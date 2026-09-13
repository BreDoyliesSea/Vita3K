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

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

#include <cpu/common.h>
#include <kernel/state.h>
#include <mem/functions.h>

#include <kernel/thread/thread_state.h>

#include <cpu/functions.h>
#include <mem/ptr.h>
#include <util/lock_and_find.h>
#include <util/log.h>

#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif

#include <chrono>
#include <thread>

#include <SDL3/SDL_mutex.h>
#include <SDL3/SDL_thread.h>

int CorenumAllocator::new_corenum() {
    const std::lock_guard<std::mutex> guard(lock);

    uint32_t size = 1;
    return alloc.allocate_from(0, size);
}

void CorenumAllocator::free_corenum(const int num) {
    const std::lock_guard<std::mutex> guard(lock);
    alloc.free(num, 1);
}

void CorenumAllocator::set_max_core_count(const std::size_t max) {
    const std::lock_guard<std::mutex> guard(lock);
    alloc.set_maximum(max);
}

// TODO implement cross platform debug thread name setter and eliminate SDL thread
struct ThreadParams {
    KernelState *kernel = nullptr;
    SceUID thid = SCE_KERNEL_ERROR_ILLEGAL_THREAD_ID;
    SDL_Semaphore *host_may_destroy_params = nullptr;
};

static int SDLCALL thread_function(void *data) {
    assert(data != nullptr);

#ifdef _WIN32
    // Reserve stack for the exception handler to run in after a guard-page hit.
    //
    // Without it, a stack overflow on a guest thread is unreportable: the handler's own
    // LOG_CRITICAL goes through fmt and spdlog, runs off the end of what is left, smashes the
    // /GS cookie on the way out, and the process dies as 0xC0000409 (stack cookie failure) with
    // nothing in the log. Five crashes were read as buffer overruns before a dump showed the
    // handler's own frames sitting on top of an EXCEPTION_STACK_OVERFLOW.
    ULONG guarantee = 64 * 1024;
    SetThreadStackGuarantee(&guarantee);
#endif

    const ThreadParams params = *static_cast<const ThreadParams *>(data);
    SDL_SignalSemaphore(params.host_may_destroy_params);
    ThreadStatePtr thread = params.kernel->get_thread(params.thid);
#ifdef TRACY_ENABLE
    if (!thread->name.empty()) {
        tracy::SetThreadName(thread->name.c_str());
    } else {
        std::string th_name = "TID:" + std::to_string(thread->id);
        tracy::SetThreadName(th_name.c_str());
    }
#endif

    thread->run_loop();
    const uint32_t r0 = read_reg(*thread->cpu, 0);
    const SceUID id = thread->id;
    const int processor_id = get_processor_id(*thread->cpu);
    // release our reference first so the erase below destroys the ThreadState before process_exit() is woken
    thread.reset();

    {
        std::lock_guard<std::mutex> lock(params.kernel->mutex);
        params.kernel->threads.erase(id);
        params.kernel->corenum_allocator.free_corenum(processor_id);
        params.kernel->thread_deleted_cond.notify_all();
    }

    return r0;
}

KernelState::KernelState()
    : debugger(*this) {
}

bool KernelState::init(MemState &mem, const CallImportFunc &call_import, bool cpu_opt) {
    corenum_allocator.set_max_core_count(MAX_CORE_COUNT);
    start_tick = rtc_get_ticks(rtc_base_ticks());
    base_tick = { rtc_base_ticks() };
    this->call_import = call_import;
    this->cpu_opt = cpu_opt;

    // Generate halt instruction (NOP + WFI)
    halt_instruction = alloc_block(mem, 4, "halt_instruction");
    const auto halt_ptr = halt_instruction.get_ptr<uint16_t>().get(mem);
    halt_ptr[0] = 0xBF00; // NOP
    halt_ptr[1] = 0xBF30; // WFI
    halt_instruction_pc = halt_instruction.get() | 1; // thumb mode pc

    return true;
}

void KernelState::load_process_param(MemState &mem, Ptr<uint32_t> ptr) {
    const SceProcessParam *param = ptr.cast<SceProcessParam>().get(mem);
    if (param->version == 0) {
        // Homebrews built with old vitasdk
        process_param = nullptr;
        return;
    }
    process_param = ptr.cast<SceProcessParam>();
    // VAR_NID(__sce_libcparam, 0xDF084DFA)
    // no memory leak because we don't allocate memory for this variable intially
    export_nids[0xDF084DFA] = process_param.get(mem)->sce_libc_param.address();
}

void KernelState::set_memory_watch(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &thread : threads) {
        auto &cpu = *thread.second->cpu;
        if (enabled != get_log_mem(cpu)) {
            if (enabled)
                set_log_mem(cpu, true);
            else
                set_log_mem(cpu, false);
        }
    }
}

void KernelState::invalidate_jit_cache(Address start, size_t length) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &[_, thread] : threads) {
        ::invalidate_jit_cache(*thread->cpu, start, length);
    }
}

ThreadStatePtr KernelState::get_thread(SceUID thread_id) {
    return lock_and_find(thread_id, threads, mutex);
}

SceUID KernelState::create_callback(const ThreadStatePtr &thread, const char *name, Ptr<SceKernelCallbackFunction> func, Ptr<void> common) {
    const CallbackPtr cb = std::make_shared<Callback>(thread, name, func, common);
    const SceUID uid = objects.add(cb, get_next_uid());
    thread->add_callback(cb);
    return uid;
}

bool KernelState::delete_callback(SceUID id) {
    return objects.remove<Callback>(id);
}

ThreadStatePtr KernelState::create_thread(MemState &mem, const char *name, Ptr<const void> entry_point) {
    return create_thread(mem, name, entry_point, SCE_KERNEL_DEFAULT_PRIORITY, SCE_KERNEL_THREAD_CPU_AFFINITY_MASK_DEFAULT, SCE_KERNEL_STACK_SIZE_USER_MAIN, nullptr);
}

// Register the thread and give it its host thread, which parks in run_loop until start().
static void spawn_host_thread(KernelState &kernel, const ThreadStatePtr &thread) {
    {
        const std::lock_guard<std::mutex> lock(kernel.mutex);
        kernel.threads.emplace(thread->id, thread);
    }

    ThreadParams params;
    params.kernel = &kernel;
    params.thid = thread->id;

    params.host_may_destroy_params = SDL_CreateSemaphore(0);
    SDL_DetachThread(SDL_CreateThread(&thread_function, thread->name.c_str(), &params));
    SDL_WaitSemaphore(params.host_may_destroy_params);
    SDL_DestroySemaphore(params.host_may_destroy_params);
}

ThreadStatePtr KernelState::create_thread(MemState &mem, const char *name, Ptr<const void> entry_point, int init_priority, SceInt32 affinity_mask, int stack_size, const SceKernelThreadOptParam *option) {
    ThreadStatePtr thread = std::make_shared<ThreadState>(get_next_uid(), *this, mem);
    if (thread->init(name, entry_point, init_priority, affinity_mask, stack_size, option) < 0)
        return nullptr;

    spawn_host_thread(*this, thread);
    return thread;
}

ThreadStatePtr KernelState::create_thread_for_savestate(MemState &mem, SceUID id, const std::string &name, Address entry_point, int priority, SceInt32 affinity_mask, Address stack, int stack_size, Address tls, const CPUContext &init_ctx, bool resume_after_load) {
    ensure_next_uid_above(id);
    ThreadStatePtr thread = std::make_shared<ThreadState>(id, *this, mem);
    if (thread->adopt_for_savestate(name, entry_point, priority, affinity_mask, stack, stack_size, tls, init_ctx) < 0)
        return nullptr;

    spawn_host_thread(*this, thread);
    if (resume_after_load) {
        thread->hold_for_savestate_resume();
        const std::lock_guard<std::mutex> lock(mutex);
        paused_threads_status[id] = ThreadStatus::run;
    }
    return thread;
}

void KernelState::ensure_next_uid_above(SceUID id) {
    SceUID current = next_uid.load();
    while (current <= id && !next_uid.compare_exchange_weak(current, id + 1)) {
    }
}

Ptr<Ptr<void>> KernelState::get_thread_tls_addr(MemState &mem, SceUID thread_id, int key) {
    Ptr<Ptr<void>> address(0);
    // magic numbers taken from decompiled source. There is 0x400 unused bytes of unknown usage
    if (key <= 0x100 && key >= 0) {
        const ThreadStatePtr thread = get_thread(thread_id);
        address = thread->tls.get_ptr<Ptr<void>>() + key;
    } else {
        LOG_ERROR("Wrong tls slot index. TID:{} index:{}", thread_id, key);
    }
    return address;
}

void KernelState::request_process_exit(int res, std::optional<AppLaunchRequest> relaunch) {
    if (process_exit_callback)
        process_exit_callback(res, std::move(relaunch));
}

void KernelState::process_exit() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto &[_, thread] : threads)
            thread->exit_delete(false);
    }

    std::unique_lock<std::mutex> lock(mutex);
    thread_deleted_cond.wait(lock, [this] { return threads.empty(); });
}

void KernelState::pause_threads() {
    const std::lock_guard<std::mutex> lock(mutex);
    for (auto &[_, thread] : threads) {
        // One reading of the status, taken under the thread's lock, both recorded and acted on.
        // Reading it twice let a thread started in between -- SceGxmDisplayQueue, which the display
        // queue's host thread starts with start() at any moment -- be stopped while recorded as
        // dormant, and resume_threads never resumed it: Gravity Rush's display queue wedged for
        // good after two quickload attempts. A waiting thread is stopped too, on its way out of its
        // call; otherwise a 10 ms delay running out mid-save sent it back into guest code while
        // guest memory was being copied or rewritten.
        paused_threads_status[thread->id] = thread->request_pause();
    }
}

bool KernelState::wait_for_threads_paused(std::chrono::milliseconds timeout, std::string *blocker) {
    // Poll rather than wait on a condition variable: the transition to suspend happens in
    // ThreadState::run_loop under the *thread's* own mutex, and there is no single place that
    // signals "all of them are now parked". A short sleep between passes is cheap next to the
    // work the caller is about to do.
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        std::string still_running;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            for (const auto &[id, thread] : threads) {
                if (!thread)
                    continue;
                // Only threads we asked to stop matter. One that was already waiting or dormant
                // when pause_threads() ran is not executing guest code and never will be until
                // it is resumed.
                const auto recorded = paused_threads_status.find(id);
                if (recorded == paused_threads_status.end() || recorded->second != ThreadStatus::run)
                    continue;
                if (thread->status == ThreadStatus::run) {
                    // Say which HLE call it is in, if any: a thread that will not stop is usually
                    // one blocked in a call that does not report its wait.
                    const uint32_t nid = thread->current_import_nid.load(std::memory_order_relaxed);
                    still_running = nid ? fmt::format("{} \"{}\" (in import {:#010x})", id, thread->name, nid)
                                        : fmt::format("{} \"{}\"", id, thread->name);
                    break;
                }
            }
        }

        if (still_running.empty())
            return true;

        if (std::chrono::steady_clock::now() >= deadline) {
            if (blocker)
                *blocker = still_running;
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KernelState::resume_threads() {
    const std::lock_guard<std::mutex> lock(mutex);
    for (auto &[id, thread] : threads) {
        // Only threads this pause stopped. Looking them up with operator[] defaulted a thread
        // created during the pause to ThreadStatus::running -- enumerator 0 -- and started it.
        const auto recorded = paused_threads_status.find(id);
        if (recorded == paused_threads_status.end())
            continue;
        // A thread that was running may have gone into a wait on its way to stopping, and one that
        // was waiting may have come out and parked. Resume whichever has parked, and leave the
        // ones still waiting where they are rather than waking them out of it.
        if (recorded->second == ThreadStatus::running || recorded->second == ThreadStatus::waiting)
            thread->resume_after_pause();
    }
    paused_threads_status.clear();
}

void KernelState::deinit(MemState &mem) {
    process_exit();
    threads.clear();

    objects.clear();

    loaded_modules.clear();
    loaded_sysmodules.clear();
    loaded_internal_sysmodules.clear();

    {
        std::lock_guard<std::mutex> lock(export_nids_mutex);
        export_nids.clear();
        export_nids_by_lib.clear();
        export_nid_owners.clear();
        func_binding_infos.clear();
        var_binding_infos.clear();
        module_uid_by_nid.clear();
    }

    corenum_allocator.alloc.reset();
    corenum_allocator.alloc.set_maximum(0);

    obj_store.clear();

    tls_address = Ptr<const void>(0);
    tls_psize = 0;
    tls_msize = 0;

    thread_event_start = Ptr<const void>(0);
    thread_event_start_arg = 0;
    thread_event_end = Ptr<const void>(0);
    thread_event_end_arg = 0;

    codec_blocks.clear();

    halt_instruction = nullptr;
    halt_instruction_pc = 0;

    process_param = nullptr;
    client_vtable = Ptr<void>(0);
    shellsvc_client = Ptr<Address>(0);
    libc_dso_handle_main = Ptr<void>(0);

    debugger.deinit();

    next_uid = 1;

    paused_threads_status.clear();
}

SceKernelModuleInfo *KernelState::find_module_by_addr(Address address) {
    const auto lock = std::lock_guard(mutex);
    for (auto &[_, mod] : loaded_modules) {
        for (auto &seg : mod->info.segments) {
            if (!seg.size)
                continue;
            if (seg.vaddr.address() <= address && address <= seg.vaddr.address() + seg.memsz) {
                return &mod->info;
            }
        }
    }
    return nullptr;
}
