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

#include <condition_variable>
#include <cstdint>
#include <list>
#include <mutex>

struct ThreadState;

// Shares the three user cores of a Vita among the guest threads the way its kernel does, as the
// VitaAutotests scheduler tests observe it. Every guest thread keeps its own host thread, but it
// only runs guest code while it holds a core. A thread gets a core when one of its mask is idle,
// or by taking it from a thread of a lower priority, which stops in the JIT and waits READY. Threads
// of equal priority run one after another, there is no time slicing. Priorities 0x40 to 0x7F wait
// in the queues of one core, a thread made ready from another core as STANDBY until that core next
// switches threads. Priorities 0x80 to 0xBF wait in one queue the cores share.
// Every method takes the scheduler's mutex, which is taken after a thread's mutex and never the
// other way round, and never while an object lock is wanted.
class Scheduler {
public:
    static constexpr int CORE_COUNT = 3;
    static constexpr int NO_CORE = -1;
    // What thread info reports as the current CPU of a thread READY in the queue the cores share
    static constexpr int SHARED_QUEUE_CPU = 0x7FFF;
    static constexpr int SHARED_AREA_PRIORITY = 0x80;

    enum class State : uint8_t {
        // Not runnable: dormant, waiting, held or deleted
        none,
        running,
        ready,
        standby,
    };

    // The scheduler's view of one thread. Guarded by the scheduler's mutex.
    struct Node {
        State state = State::none;
        // The core it runs on, waits for as STANDBY or READY in that core's queue, or NO_CORE
        int core = NO_CORE;
        int last_core = NO_CORE;
        // The thread's priority and affinity mask, which it passes on whenever it changes them
        int priority = 0;
        int32_t affinity_mask = 0;
        // Set when a hold wants the thread to leave its gate, and when it is deleted
        bool kicked = false;
        bool removed = false;
        std::condition_variable cv;
    };

    struct Info {
        State state;
        int current_cpu;
        int last_cpu;
    };

    void set_enabled(bool value) { enabled = value; }
    bool is_enabled() const { return enabled; }

    // The guest thread the calling host thread runs, or null on a host thread
    static ThreadState *current();
    static void set_current(ThreadState *thread);

    // Makes a thread runnable, placing it by priority and affinity, from the core of from (a guest
    // thread), or from no core for a timeout, an interrupt or a host thread. Does nothing to a thread
    // that is runnable already.
    void make_ready(ThreadState &thread, ThreadState *from);
    // The thread stops running or being runnable: it waits, ends or is held. Its core goes to the next
    // thread.
    void block(ThreadState &thread);
    // The thread is deleted: like block, and its gate returns for good.
    void leave(ThreadState &thread);
    // Blocks the calling guest thread until it holds a core. Returns false if it should go back to
    // its holds or its ending instead of running.
    bool wait_for_core(ThreadState &thread);
    // Makes wait_for_core return false once, so the thread looks at its holds
    void kick(ThreadState &thread);
    // The thread's priority changed: moves it between queues and cores as a Vita would
    void change_priority(ThreadState &thread, int priority);
    // The thread's affinity mask changed: moves it off a core the mask leaves out
    void change_affinity(ThreadState &thread, int32_t mask, ThreadState *from);
    Info info(ThreadState &thread);
    // Forgets every thread, for the end of the process
    void reset();

private:
    struct Core {
        ThreadState *running = nullptr;
        // Threads READY on this core, by priority, FIFO within one
        std::list<ThreadState *> ready;
        // Threads handed to this core from another, in arrival order
        std::list<ThreadState *> standby;
    };

    static unsigned cores_of(const ThreadState &thread);
    static bool in_shared_area(const ThreadState &thread);
    static void insert_by_priority(std::list<ThreadState *> &queue, ThreadState &thread, bool at_head);
    void remove_from_queues(ThreadState &thread);
    void run_on(int core, ThreadState &thread);
    void take_core(int core, ThreadState &thread);
    void give_up_core(ThreadState &thread);
    void place(ThreadState &thread, int from_core, bool preempted);
    void schedule_core(int core);
    ThreadState *best_waiting(int core, bool with_standby);
    // The core of the mask running the lowest priority, preferring the thread's last core on ties
    int lowest_core(const ThreadState &thread, unsigned mask) const;
    void preempt_if_better(ThreadState &thread);

    bool enabled = true;
    std::mutex mutex;
    Core cores[CORE_COUNT];
    std::list<ThreadState *> shared_ready;
};
