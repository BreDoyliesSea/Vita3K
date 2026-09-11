// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#pragma once

#include <cpu/state.h>
#include <kernel/callback.h>
#include <kernel/thread/wait_queue.h>
#include <kernel/types.h>
#include <mem/block.h>
#include <mem/ptr.h>

#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <optional>
#include <string>

struct CPUContext;

struct ThreadState;
struct ThreadParams;
struct KernelState;

typedef std::unique_ptr<CPUState, std::function<void(CPUState *)>> CPUStatePtr;
typedef std::function<void(CPUState &, uint32_t, SceUID)> CallImport;
typedef std::function<std::string(Address)> ResolveNIDName;

// Values are what sceKernelGetThreadInfo reports
enum class ThreadStatus : SceUInt32 {
    running = SCE_KERNEL_THREAD_STATUS_RUNNING,
    waiting = SCE_KERNEL_THREAD_STATUS_WAITING, // Waiting to be awaken by sync object or operation
    dormant = SCE_KERNEL_THREAD_STATUS_DORMANT, // Waiting for a job
    suspended = SCE_KERNEL_THREAD_STATUS_SUSPENDED, // Suspended by debugger
};

struct ThreadState {
    std::mutex mutex;
    std::string name;
    SceUID id;
    Address entry_point;

    Block stack;
    int stack_size;
    Block tls;

    int priority;
    SceInt32 affinity_mask;
    uint64_t start_tick;
    uint64_t last_vblank_waited;

    CPUStatePtr cpu;
    ThreadStatus status = ThreadStatus::dormant;
    // What the thread waits on while waiting, empty otherwise
    WaitTarget wait_target;

    // NID of the HLE import this thread is currently executing, 0 when it is running guest code,
    // and the ABI argument registers it was called with. Maintained by call_import(). A thread in
    // ThreadStatus::waiting is parked inside one of these, in a host C++ frame that no snapshot of
    // guest memory can describe, so together these are what a restore would need in order to
    // re-enter the call rather than reconstruct the frame.
    //
    // The NID is atomic because a savestate reads it from another thread that may still be
    // running. The arguments are plain: they are only ever read for a thread already parked in
    // ThreadStatus::waiting, which by definition is not writing them.
    std::atomic<uint32_t> current_import_nid{ 0 };
    uint32_t current_import_args[4]{};

    std::condition_variable status_cond;
    uint32_t returned_value = 0;

    ThreadState() = delete;
    explicit ThreadState(SceUID id, KernelState &kernel, MemState &mem);

    int init(const char *name, Ptr<const void> entry_point, int init_priority, SceInt32 affinity_mask, int stack_size, const SceKernelThreadOptParam *option);
    int start(SceSize arglen, const Ptr<void> argp, bool run_entry_callback = false);
    void exit(SceInt32 status);
    void exit_delete(bool exit = true);

    void update_status(ThreadStatus status, std::optional<ThreadStatus> expected = std::nullopt);
    Address stack_top() const;

    void run_loop();

    // this function must be called from the thread itself (inside a svc call)
    uint32_t run_callback(Address callback_address, const std::vector<uint32_t> &args);

    // this function is called from another thread when this one is dormant
    // it is only used for module loading and gxm display queue right now
    // args and argp are passed to thread->start as is
    uint32_t run_guest_function(Address callback_address, SceSize args = 0, const Ptr<void> argp = Ptr<void>{});

    // Blocks this thread until the deadline passes.
    [[nodiscard]] WaitResult delay_until(Deadline deadline, bool callbacks);
    // Blocks this thread until a signal is sent to it.
    [[nodiscard]] WaitResult wait_for_signal(bool callbacks);
    // Sends a signal to this thread. Fails if the previous one was not consumed yet.
    SceInt32 send_signal();
    // Blocks waiter until this thread becomes dormant, then writes its exit status to exit_status.
    [[nodiscard]] WaitResult wait_for_thread_end(const ThreadStatePtr &waiter, SceInt32 *exit_status, bool callbacks);

    // Waits on target until woken by wake(), the thread exits or is deleted, or the deadline passes.
    // With callbacks, it also returns after running callbacks that were notified meanwhile.
    // A stale wake or callbacks can end it early, so callers must recheck their condition.
    [[nodiscard]] WaitResult wait(WaitTarget target, Deadline deadline, bool callbacks);
    // Wakes this thread from wait().
    void wake();

    // Runs the notified callbacks of this thread and returns how many ran. Called by the thread itself.
    SceUInt32 process_callbacks();
    // Tells this thread that one of its callbacks was notified, so a wait with callbacks runs it.
    void notify_callbacks();
    // Adds a callback this thread created. Called by the thread itself.
    void add_callback(const CallbackPtr &cb);

    // sceKernelSuspendThreadForVM / ResumeThreadForVM, with which Mono's garbage collector stops the
    // other threads, reads their registers and lets them go. Unlike suspend(), suspend_for_vm()
    // returns only once the thread has stopped running guest code (false if it had not within
    // timeout_ms), and the calls nest: resume_for_vm() undoes exactly one of them, even one still
    // taking effect, and never ends a suspend that came from anywhere else. Returns false if the
    // thread was not suspended.
    bool suspend_for_vm(int timeout_ms);
    bool resume_for_vm();

    void suspend();
    void resume(bool step = false);
    std::string log_stack_traceback() const;

    // --- pausing and savestates -----------------------------------------------------------------
    // Depth of run_loop frames: 1 for a thread at its top level, more inside a callback.
    int nesting_level() const {
        return call_level;
    }

    // For KernelState::pause_threads: stop this thread whether it is running guest code or waiting in
    // an HLE call (then it parks the moment the call returns), and return the status it had -- read
    // once, under the thread's lock, so the status recorded and the stop applied always agree.
    ThreadStatus request_pause();

    // End of a pause for a thread that was waiting when it began: resume it if its call returned
    // and it parked, otherwise cancel the request. Returns true if it was resumed.
    bool resume_after_pause();

    // Savestate load. If the thread is still waiting inside the import `nid`, arrange that when the
    // call returns it waits for the load to finish -- the load holds MemState::savestate_lock
    // exclusively -- and then continues from `ctx` rather than from wherever the call left it.
    // Returns false if it is no longer in that call.
    bool continue_from_on_return(uint32_t nid, const CPUContext &ctx);

    // True while the thread is waiting on the load after continue_from_on_return.
    bool held_for_savestate() const {
        return held_for_load.load(std::memory_order_acquire);
    }

    // How a savestate load takes this thread out of the HLE wait it is parked in. Every blocking
    // call registers one for as long as it is parked (ScopedWait, below), so the load needs no
    // knowledge of the individual calls.
    struct WaitRelease {
        // Ends the wait without it being satisfied, and wakes the thread. Empty for a wait that
        // cannot be woken and ends on its own (sceAudioOutOutput).
        std::function<void()> release;
        // Whether making the call again from its start is equivalent to staying in it. A save
        // refuses a moment when a thread is in a wait that is not.
        bool restartable = true;
    };
    void enter_wait(WaitRelease how);
    void leave_wait();
    std::optional<WaitRelease> current_wait() const;

    // Set once a savestate load has armed this thread (continue_from_on_return) and is ending its
    // wait. A wait that would otherwise go back to sleep, or run guest callbacks, on waking checks
    // this and returns instead.
    bool wait_abandoned() const {
        return abandon_wait.load(std::memory_order_acquire);
    }

private:
    // Whether the thread is exiting or being deleted. Called with mutex held.
    bool exiting() const { return exit_requested || delete_requested; }

    void push_arguments(const std::vector<uint32_t> &args);
    void dispatch_abort(CPUState &cpu);

    KernelState &kernel;

    CPUContext init_cpu_ctx;
    // sceKernelExitThread (or top-level guest function return): park at dormant, thread reusable via start() / run_guest_function().
    bool exit_requested = false;
    // sceKernelExitDeleteThread (or external kill): will return from top-level run_loop(), then host thread joins.
    bool delete_requested = false;
    // Set by suspend(), consumed in run_loop() to transition to ThreadStatus::suspended.
    bool suspend_requested = false;
    // Set by continue_from_on_return(), consumed in run_loop() when the call returns.
    std::optional<CPUContext> context_on_return;
    std::atomic<bool> held_for_load{ false };
    std::atomic<bool> abandon_wait{ false };
    // Guards wait_release only, and is never held while calling anything, so a wait can register
    // under whatever locks it already holds.
    mutable std::mutex wait_release_mutex;
    std::optional<WaitRelease> wait_release;
    // Single stepping mode.
    bool single_stepping = false;

    // Number of active run_loop frames. The top-level host thread keeps one
    // frame alive (run_loop()) while parked dormant; callbacks add nested frames.
    int call_level = 0;
    // So the warning about runaway nesting is logged once, not on every frame past the threshold.
    bool deep_nesting_reported = false;

    // when calling sceKernelStartThread
    bool run_start_callback = false;
    // when calling sceKernelExitThread or sceKernelExitDeleteThread
    bool run_end_callback = false;

    // Outstanding suspend_for_vm() calls; while any are, run_loop parks rather than run guest code.
    int vm_suspend_count = 0;
    // Whether the current park is for the VM alone, and so ended by the last resume_for_vm().
    bool parked_for_vm = false;
    // True only while run() or step() is executing guest code.
    std::atomic<bool> in_guest_code{ false };

    MemState &mem;

    // A sceKernelSendSignal is pending for this thread.
    bool signal_pending = false;
    // Set by wake() and consumed by the next wait().
    bool wake_pending = false;
    // Set by notify_callbacks() and cleared when the callbacks run.
    bool callbacks_pending = false;
    // Set while the thread runs its callbacks. They don't nest.
    bool is_processing_callbacks = false;
    // Callbacks this thread created, in creation order. The kernel owns them. Only this thread touches the list.
    std::list<std::weak_ptr<Callback>> callbacks;

    // Notified under mutex whenever a condition a wait may be blocked on changes.
    std::condition_variable wait_cv;

    struct EndWaitEntry {
        // Where to write the exit status, or null
        SceInt32 *exit_status;
    };

    // Guards end_waiters. Taken after mutex when both are needed.
    std::mutex end_waiters_mutex;
    // Threads blocked in sceKernelWaitThreadEnd on this one.
    WaitQueue<EndWaitEntry> end_waiters;
};

// Registers how to release a wait for as long as it is in scope (see ThreadState::WaitRelease).
struct ScopedWait {
    ScopedWait(ThreadState &thread, ThreadState::WaitRelease how)
        : thread(thread) {
        thread.enter_wait(std::move(how));
    }
    ~ScopedWait() {
        thread.leave_wait();
    }
    ScopedWait(const ScopedWait &) = delete;
    ScopedWait &operator=(const ScopedWait &) = delete;

private:
    ThreadState &thread;
};

typedef std::shared_ptr<ThreadState> ThreadStatePtr;
