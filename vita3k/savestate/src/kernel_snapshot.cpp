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

#include "kernel_snapshot.h"

#include <cpu/functions.h>
#include <emuenv/state.h>
#include <gxm/state.h>
#include <kernel/callback.h>
#include <kernel/state.h>
#include <kernel/sync_primitives.h>
#include <kernel/thread/thread_state.h>
#include <mem/functions.h>
#include <mem/state.h>
#include <nids/functions.h>
#include <renderer/state.h>
#include <util/log.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <span>
#include <thread>
#include <type_traits>

namespace savestate::kernel {

namespace {

constexpr SceUInt32 WAITTYPE_CB_BIT = 0x80000000U;

// Host waits, which report WAITING with no kernel wait, that a thread can be stopped in and make
// again from its stub: each one blocks before its call has done anything the guest can see.
constexpr const char *RESTARTABLE_HOST_WAITS[] = { "sceGxmDisplayQueueAddEntry", "sceGxmNotificationWait", "sceAudioOutOutput" };

struct Writer {
    std::vector<uint8_t> &out;

    template <typename T>
    void put(const T &value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
        out.insert(out.end(), bytes, bytes + sizeof(T));
    }
    void put_string(const std::string &s) {
        put<uint32_t>(static_cast<uint32_t>(s.size()));
        out.insert(out.end(), s.begin(), s.end());
    }
    void put_bytes(const std::vector<uint8_t> &bytes) {
        put<uint32_t>(static_cast<uint32_t>(bytes.size()));
        out.insert(out.end(), bytes.begin(), bytes.end());
    }
};

struct Reader {
    const uint8_t *data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    template <typename T>
    bool get(T &value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (pos + sizeof(T) > size)
            return false;
        std::memcpy(&value, data + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }
    bool get_string(std::string &s) {
        uint32_t n = 0;
        if (!get(n) || pos + n > size)
            return false;
        s.assign(reinterpret_cast<const char *>(data + pos), n);
        pos += n;
        return true;
    }
    bool get_bytes(std::vector<uint8_t> &bytes) {
        uint32_t n = 0;
        if (!get(n) || pos + n > size)
            return false;
        bytes.assign(data + pos, data + pos + n);
        pos += n;
        return true;
    }
};

// The threads that exist for the guest, taken out of the table so their own locks can be taken
std::vector<ThreadStatePtr> live_threads(KernelState &kernel) {
    std::vector<ThreadStatePtr> threads;
    const std::lock_guard<std::mutex> lock(kernel.mutex);
    for (const auto &[_, thread] : kernel.threads) {
        if (thread && !thread->is_deleted())
            threads.push_back(thread);
    }
    return threads;
}

// The thread an operation acts as when it needs one and no thread in particular: a read lock, a pipe
// read or write
ThreadStatePtr any_thread(KernelState &kernel) {
    const std::vector<ThreadStatePtr> threads = live_threads(kernel);
    return threads.empty() ? nullptr : threads.front();
}

bool quiet(const SceKernelThreadInfo &info) {
    return info.status == SCE_KERNEL_THREAD_STATUS_DORMANT || info.status == SCE_KERNEL_THREAD_STATUS_SUSPENDED
        || info.status == SCE_KERNEL_THREAD_STATUS_WAITING;
}

bool stopped(const SceKernelThreadInfo &info) {
    return info.status == SCE_KERNEL_THREAD_STATUS_DORMANT || info.status == SCE_KERNEL_THREAD_STATUS_SUSPENDED;
}

// Waits until done holds for every thread, for at most timeout_ms. Returns the first thread it
// doesn't hold for then, or null.
template <typename Done>
ThreadStatePtr wait_for_all(KernelState &kernel, int timeout_ms, Done done) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        ThreadStatePtr blocker;
        for (const ThreadStatePtr &thread : live_threads(kernel)) {
            if (!done(thread->info())) {
                blocker = thread;
                break;
            }
        }
        if (!blocker || std::chrono::steady_clock::now() >= deadline)
            return blocker;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

// The import a thread blocked in a call is in: the NID follows the mov pc, lr its PC is at
const char *call_name(MemState &mem, const CPUContext &ctx) {
    const Address at = ctx.get_pc() + 4;
    if (!Ptr<uint32_t>(at).valid(mem))
        return nullptr;
    return import_name(*Ptr<uint32_t>(at).get(mem));
}

const char *status_name(SceUInt32 status) {
    switch (status) {
    case SCE_KERNEL_THREAD_STATUS_RUNNING: return "running";
    case SCE_KERNEL_THREAD_STATUS_READY: return "ready";
    case SCE_KERNEL_THREAD_STATUS_WAITING: return "waiting";
    case SCE_KERNEL_THREAD_STATUS_DORMANT: return "dormant";
    case SCE_KERNEL_THREAD_STATUS_SUSPENDED: return "suspended";
    default: return "in another state";
    }
}

template <KernelObjectClass T>
std::vector<std::shared_ptr<T>> objects_of(const KernelState &kernel) {
    std::vector<SceUID> uids;
    kernel.objects.for_each<T>([&](T &obj) { uids.push_back(obj.uid); });
    std::vector<std::shared_ptr<T>> objects;
    for (const SceUID uid : uids) {
        if (std::shared_ptr<T> obj = kernel.objects.find<T>(uid))
            objects.push_back(std::move(obj));
    }
    return objects;
}

template <KernelObjectClass T>
void delete_all(KernelState &kernel) {
    std::vector<SceUID> uids;
    kernel.objects.for_each<T>([&](T &obj) { uids.push_back(obj.uid); });
    for (const SceUID uid : uids)
        kernel.objects.remove<T>(uid, 0);
}

// Records one object of class T. The recipe comes first, then what its info reports.
template <KernelObjectClass T>
ObjectRecord record_of(const T &obj, const std::function<void(Writer &)> &fill) {
    ObjectRecord record{ static_cast<uint32_t>(T::uid_class), obj.uid, {} };
    Writer w{ record.data };
    fill(w);
    return record;
}

// What a message pipe holds, read without taking it, through a scratch block of guest memory
std::vector<uint8_t> pipe_contents(EmuEnvState &emuenv, MsgPipe &pipe, SceSize used) {
    std::vector<uint8_t> bytes;
    const ThreadStatePtr thread = any_thread(emuenv.kernel);
    if (used == 0 || !thread)
        return bytes;
    const Address scratch = alloc(emuenv.mem, used, "savestate pipe read");
    if (!scratch)
        return bytes;
    const MsgPipeVector piece{ Ptr<void>(scratch), used };
    SceSize got = 0;
    pipe.try_receive(thread, emuenv.mem, SCE_KERNEL_MSG_PIPE_MODE_ASAP | SCE_KERNEL_MSG_PIPE_MODE_DONT_REMOVE, std::span(&piece, 1), &got);
    const uint8_t *const at = Ptr<uint8_t>(scratch).get(emuenv.mem);
    bytes.assign(at, at + std::min(got, used));
    free(emuenv.mem, scratch);
    return bytes;
}

void put_back_pipe_contents(EmuEnvState &emuenv, MsgPipe &pipe, const std::vector<uint8_t> &bytes) {
    const ThreadStatePtr thread = any_thread(emuenv.kernel);
    if (bytes.empty() || !thread)
        return;
    const auto size = static_cast<SceSize>(bytes.size());
    const Address scratch = alloc(emuenv.mem, size, "savestate pipe write");
    if (!scratch)
        return;
    std::memcpy(Ptr<uint8_t>(scratch).get(emuenv.mem), bytes.data(), bytes.size());
    const MsgPipeVector piece{ Ptr<void>(scratch), size };
    SceSize moved = 0;
    pipe.try_send(thread, emuenv.mem, SCE_KERNEL_MSG_PIPE_MODE_ASAP, std::span(&piece, 1), &moved);
    if (moved != size)
        LOG_WARN("Savestate: message pipe \"{}\" ({}) took {} of its {} bytes back", pipe.name, pipe.uid, moved, size);
    free(emuenv.mem, scratch);
}

// The threads the last load had making their call again, in the order to let them go
std::vector<SceUID> release_order;

} // namespace

std::string why_not_at_a_call(const WordReader &read, const CPUContext &ctx) {
    // Import stubs are ARM: svc #0, mov pc, lr, then the NID. A thread inside the call has its PC
    // just past the svc, and its argument registers are untouched until the call returns.
    const uint32_t pc = ctx.get_pc();
    if (ctx.cpsr & 0x20)
        return fmt::format("its PC {} is in Thumb code, not an import stub", log_hex(pc));
    const std::optional<uint32_t> before = pc < 4 ? std::nullopt : read(pc - 4);
    const std::optional<uint32_t> at = read(pc);
    if (!before || !at)
        return fmt::format("its PC {} is not in mapped memory", log_hex(pc));
    constexpr uint32_t SVC_0 = 0xEF000000U;
    constexpr uint32_t MOV_PC_LR = 0xE1A0F00EU;
    if (*before != SVC_0 || *at != MOV_PC_LR)
        return fmt::format("its PC {} is not just past an import stub's svc ({} at PC-4, {} at PC)",
            log_hex(pc), log_hex(*before), log_hex(*at));
    return {};
}

std::string quiesce(EmuEnvState &emuenv, int timeout_ms) {
    emuenv.kernel.suspend_threads();
    if (const ThreadStatePtr blocker = wait_for_all(emuenv.kernel, timeout_ms, quiet)) {
        const SceKernelThreadInfo info = blocker->info();
        return fmt::format("thread {} \"{}\" would not stop (it is {})", blocker->id, blocker->name, status_name(info.status));
    }
    return {};
}

void release_threads(EmuEnvState &emuenv) {
    KernelState &kernel = emuenv.kernel;
    for (const SceUID id : std::exchange(release_order, {})) {
        const ThreadStatePtr thread = kernel.get_thread(id);
        if (!thread)
            continue;
        thread->resume();
        // Until it blocks in its call again, or gets what it waits for at once
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        while (std::chrono::steady_clock::now() < deadline) {
            const SceUInt32 status = thread->info().status;
            if (status != SCE_KERNEL_THREAD_STATUS_RUNNING && status != SCE_KERNEL_THREAD_STATUS_READY && status != SCE_KERNEL_THREAD_STATUS_SUSPENDED)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    kernel.resume_suspended_threads();
}

Result capture(EmuEnvState &emuenv, Snapshot &snapshot) {
    KernelState &kernel = emuenv.kernel;
    MemState &mem = emuenv.mem;
    const WordReader session_word = [&mem](Address address) -> std::optional<uint32_t> {
        if (!Ptr<uint32_t>(address).valid(mem))
            return std::nullopt;
        return *Ptr<uint32_t>(address).get(mem);
    };

    for (const ThreadStatePtr &thread : live_threads(kernel)) {
        const SceKernelThreadInfo info = thread->info();
        ThreadRecord r;
        r.id = thread->id;
        r.name = thread->name;
        r.entry_point = thread->entry_point;
        r.attr = thread->attr;
        r.stack_size = thread->stack_size;
        r.stack = thread->stack.get();
        r.tls = thread->tls.get();
        r.init_priority = thread->init_priority;
        r.init_affinity_mask = thread->init_affinity_mask;
        r.init_cpu_ctx = thread->init_cpu_ctx;
        r.priority = info.currentPriority;
        r.affinity_mask = info.currentCpuAffinityMask;
        r.vm_suspends = thread->vm_suspend_count();
        r.ctx = save_context(*thread->cpu);
        r.tpidrurw = read_tpidrurw(*thread->cpu);
        r.wait_type = info.waitType & ~WAITTYPE_CB_BIT;

        // A thread inside a callback has host frames around it that no state describes
        const bool in_callback = thread->call_depth() > 1;
        switch (info.status) {
        case SCE_KERNEL_THREAD_STATUS_DORMANT:
            r.resume = Resume::dormant;
            break;
        case SCE_KERNEL_THREAD_STATUS_SUSPENDED:
            if (in_callback)
                return Result::retry(fmt::format("thread {} \"{}\" is inside a callback", r.id, r.name));
            r.resume = Resume::registers;
            break;
        case SCE_KERNEL_THREAD_STATUS_WAITING: {
            if (in_callback)
                return Result::retry(fmt::format("thread {} \"{}\" is waiting inside a callback", r.id, r.name));
            const char *const call = call_name(mem, r.ctx);
            if (const std::string why = why_not_at_a_call(session_word, r.ctx); !why.empty())
                return Result::retry(fmt::format("thread {} \"{}\" is in {} but {}", r.id, r.name, call ? call : "a call", why));
            if (r.wait_type == 0 && std::ranges::none_of(RESTARTABLE_HOST_WAITS, [&](const char *name) { return call && std::strcmp(call, name) == 0; }))
                return Result::retry(fmt::format("thread {} \"{}\" is in {}, which a load could not make again", r.id, r.name, call ? call : "an unknown call"));
            r.resume = Resume::restart_call;
            if (const auto left = thread->wait_time_left())
                r.time_left_us = left->count();
            r.wait_sequence = thread->wait_sequence();
            break;
        }
        default:
            return Result::retry(fmt::format("thread {} \"{}\" is {}", r.id, r.name, status_name(info.status)));
        }
        snapshot.threads.push_back(std::move(r));
    }

    std::vector<ObjectRecord> &objects = snapshot.objects;
    for (const auto &obj : objects_of<Semaphore>(kernel)) {
        SceKernelSemaInfo info{};
        if (obj->info(info) != SCE_KERNEL_OK)
            continue;
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(obj->init_val);
            w.put(obj->max);
            w.put(info.currentCount);
        }));
    }
    for (const auto &obj : objects_of<EventFlag>(kernel)) {
        SceKernelEventFlagInfo info{};
        if (obj->info(info) != SCE_KERNEL_OK)
            continue;
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(obj->init_pattern);
            w.put(info.currentPattern);
        }));
    }
    for (const auto &obj : objects_of<SimpleEvent>(kernel)) {
        SceKernelEventInfo info{};
        if (obj->info(info) != SCE_KERNEL_OK)
            continue;
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(info.eventPattern);
            w.put(info.userData);
        }));
    }
    for (const auto &obj : objects_of<Timer>(kernel)) {
        SceKernelTimerInfo timer{};
        SceKernelEventInfo event{};
        if (obj->timer_info(timer) != SCE_KERNEL_OK || obj->info(event) != SCE_KERNEL_OK)
            continue;
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(timer.active);
            w.put(timer.currentTime);
            w.put(timer.schedule);
            w.put(timer.interval);
            w.put(timer.type);
            w.put(timer.repeat);
            w.put(event.eventPattern);
            w.put(event.userData);
        }));
    }
    for (const auto &obj : objects_of<HeavyMutex>(kernel)) {
        SceKernelMutexInfo info{};
        if (obj->info(info) != SCE_KERNEL_OK)
            continue;
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(obj->init_count);
            w.put(info.currentCount);
            w.put(info.currentOwnerId);
            w.put<uint8_t>(0);
        }));
    }
    for (const auto &obj : objects_of<LwMutex>(kernel)) {
        SceKernelLwMutexInfo info{};
        if (obj->info(info) != SCE_KERNEL_OK)
            continue;
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(obj->init_count);
            w.put<uint32_t>(obj->workarea.address());
            w.put(info.currentCount);
            w.put(info.currentOwnerId);
            w.put<uint8_t>(0);
        }));
    }
    // A condition variable keeps the mutex it was made with after the guest deleted the mutex, so
    // the mutex is made again too, then deleted once the condition variable has it
    const auto put_deleted_mutex = [&](const Mutex &mutex, bool lightweight) {
        ObjectRecord record{ static_cast<uint32_t>(lightweight ? UidClass::lw_mutex : UidClass::mutex), mutex.uid, {} };
        Writer w{ record.data };
        w.put(mutex.attr);
        w.put_string(mutex.name);
        w.put(mutex.init_count);
        if (lightweight)
            w.put<uint32_t>(mutex.workarea.address());
        w.put<SceInt32>(0);
        w.put<SceUID>(0);
        w.put<uint8_t>(1);
        objects.push_back(std::move(record));
    };
    std::set<SceUID> deleted_mutexes;
    for (const auto &obj : objects_of<HeavyCond>(kernel)) {
        if (kernel.objects.find<HeavyMutex>(obj->associated_mutex->uid) != obj->associated_mutex && deleted_mutexes.insert(obj->associated_mutex->uid).second)
            put_deleted_mutex(*obj->associated_mutex, false);
    }
    for (const auto &obj : objects_of<LwCond>(kernel)) {
        if (kernel.objects.find<LwMutex>(obj->associated_mutex->uid) != obj->associated_mutex && deleted_mutexes.insert(obj->associated_mutex->uid).second)
            put_deleted_mutex(*obj->associated_mutex, true);
    }
    for (const auto &obj : objects_of<HeavyCond>(kernel)) {
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put<SceUID>(obj->associated_mutex->uid);
        }));
    }
    for (const auto &obj : objects_of<LwCond>(kernel)) {
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put<SceUID>(obj->associated_mutex->uid);
            w.put<uint32_t>(obj->workarea.address());
        }));
    }
    for (const auto &obj : objects_of<RWLock>(kernel)) {
        SceKernelRWLockInfo info{};
        if (obj->info(info) != SCE_KERNEL_OK)
            continue;
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(info.lockCount);
            w.put(info.writeOwnerId);
        }));
    }
    for (const auto &obj : objects_of<MsgPipe>(kernel)) {
        SceKernelMppInfo info{};
        if (obj->info(info) != SCE_KERNEL_OK)
            continue;
        // Only the bytes in the pipe itself: a sender waiting with the rest makes its call again
        const std::vector<uint8_t> contents = pipe_contents(emuenv, *obj, static_cast<SceSize>(info.bufSize - info.freeSize));
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->attr);
            w.put_string(obj->name);
            w.put(info.bufSize);
            w.put_bytes(contents);
        }));
    }
    for (const auto &obj : objects_of<Callback>(kernel)) {
        objects.push_back(record_of(*obj, [&](Writer &w) {
            w.put(obj->thread_id);
            w.put_string(obj->name);
            w.put<uint32_t>(obj->cb_func.address());
            w.put<uint32_t>(obj->userdata.address());
            w.put<uint32_t>(obj->get_num_notifications());
            w.put<SceInt32>(obj->get_notify_arg());
            w.put<SceUID>(obj->get_notifier_id());
        }));
    }
    return Result::ok();
}

void write(const Snapshot &snapshot, std::vector<uint8_t> &threads, std::vector<uint8_t> &objects) {
    Writer t{ threads };
    t.put<uint32_t>(static_cast<uint32_t>(snapshot.threads.size()));
    for (const ThreadRecord &r : snapshot.threads) {
        t.put(r.id);
        t.put_string(r.name);
        t.put(r.entry_point);
        t.put(r.attr);
        t.put(r.stack_size);
        t.put(r.stack);
        t.put(r.tls);
        t.put(r.init_priority);
        t.put(r.init_affinity_mask);
        t.put(r.init_cpu_ctx);
        t.put(r.resume);
        t.put(r.wait_type);
        t.put(r.priority);
        t.put(r.affinity_mask);
        t.put(r.vm_suspends);
        t.put(r.ctx);
        t.put(r.tpidrurw);
        t.put(r.time_left_us);
        t.put(r.wait_sequence);
    }
    Writer o{ objects };
    o.put<uint32_t>(static_cast<uint32_t>(snapshot.objects.size()));
    for (const ObjectRecord &r : snapshot.objects) {
        o.put(r.uid_class);
        o.put(r.uid);
        o.put_bytes(r.data);
    }
}

bool read(const uint8_t *threads, size_t threads_size, const uint8_t *objects, size_t objects_size, Snapshot &snapshot) {
    Reader t{ threads, threads_size };
    uint32_t count = 0;
    if (!t.get(count))
        return false;
    for (uint32_t i = 0; i < count; i++) {
        ThreadRecord r;
        if (!t.get(r.id) || !t.get_string(r.name) || !t.get(r.entry_point) || !t.get(r.attr) || !t.get(r.stack_size) || !t.get(r.stack)
            || !t.get(r.tls) || !t.get(r.init_priority) || !t.get(r.init_affinity_mask) || !t.get(r.init_cpu_ctx) || !t.get(r.resume)
            || !t.get(r.wait_type) || !t.get(r.priority) || !t.get(r.affinity_mask) || !t.get(r.vm_suspends) || !t.get(r.ctx) || !t.get(r.tpidrurw)
            || !t.get(r.time_left_us) || !t.get(r.wait_sequence))
            return false;
        snapshot.threads.push_back(std::move(r));
    }
    Reader o{ objects, objects_size };
    if (!o.get(count))
        return false;
    for (uint32_t i = 0; i < count; i++) {
        ObjectRecord r;
        if (!o.get(r.uid_class) || !o.get(r.uid) || !o.get_bytes(r.data))
            return false;
        snapshot.objects.push_back(std::move(r));
    }
    return true;
}

Result plan(EmuEnvState &emuenv, const Snapshot &snapshot, const WordReader &state_word, LoadPlan &plan) {
    std::map<SceUID, ThreadStatePtr> session;
    for (const ThreadStatePtr &thread : live_threads(emuenv.kernel))
        session.emplace(thread->id, thread);
    const SceUID display_thread = emuenv.gxm.display_queue_thread;

    std::set<SceUID> in_state;
    for (const ThreadRecord &r : snapshot.threads) {
        in_state.insert(r.id);
        if (r.resume == Resume::restart_call) {
            // Checked against the state's memory, not this session's, which can have other code
            // there or nothing at all
            if (const std::string why = why_not_at_a_call(state_word, r.ctx); !why.empty())
                return Result::fail(fmt::format("thread {} \"{}\" cannot make its call again: {}", r.id, r.name, why));
        }
        const auto it = session.find(r.id);
        if (it == session.end()) {
            plan.create.push_back(r.id);
            continue;
        }
        const ThreadStatePtr &thread = it->second;
        const bool same = thread->name == r.name && thread->entry_point == r.entry_point && thread->stack.get() == r.stack
            && thread->stack_size == r.stack_size && thread->tls.get() == r.tls && thread->init_priority == r.init_priority
            && thread->init_affinity_mask == r.init_affinity_mask;
        // A thread the state has dormant has to be dormant here too: nothing but ending it gets it
        // there. Inside a callback, its host frames are the session's.
        const bool fits = same && thread->call_depth() <= 1 && (r.resume != Resume::dormant || thread->get_status() == ThreadStatus::dormant);
        if (fits) {
            plan.keep.push_back(r.id);
            continue;
        }
        // The host's display thread drives this one, and would go on driving the one it has
        if (r.id == display_thread) {
            if (same)
                return Result::retry("the display queue thread is busy");
            return Result::fail("the display queue thread is not the one the state has");
        }
        plan.remove.push_back(r.id);
        plan.create.push_back(r.id);
    }
    for (const auto &[id, thread] : session) {
        if (in_state.contains(id))
            continue;
        if (id == display_thread)
            return Result::fail("the state has no display queue thread");
        plan.remove.push_back(id);
    }
    return Result::ok();
}

void stop_session(EmuEnvState &emuenv, const LoadPlan &plan) {
    KernelState &kernel = emuenv.kernel;

    // Every call returns: the kernel's waits through the threads, the host's display queue and
    // notification waits through GXM, and an audio output by itself
    emuenv.gxm.savestate_abandon = true;
    emuenv.gxm.display_queue.wake_holders();
    {
        const std::lock_guard<std::mutex> lock(emuenv.renderer->notification_mutex);
        emuenv.renderer->notification_ready.notify_all();
    }
    for (const ThreadStatePtr &thread : live_threads(kernel))
        thread->abandon_wait(true);
    if (const ThreadStatePtr blocker = wait_for_all(kernel, 2000, stopped))
        LOG_ERROR("Savestate: thread {} \"{}\" did not get out of its call; going on", blocker->id, blocker->name);

    // The threads the state doesn't have as they are. Each has to be gone before one is made under
    // its ID: its host thread takes it out of the table by ID when it ends.
    const std::set<SceUID> going(plan.remove.begin(), plan.remove.end());
    for (const SceUID id : going) {
        if (const ThreadStatePtr thread = kernel.get_thread(id))
            thread->remove();
    }
    {
        std::unique_lock<std::mutex> lock(kernel.mutex);
        const bool gone = kernel.thread_deleted_cond.wait_for(lock, std::chrono::seconds(2), [&] {
            return std::ranges::none_of(going, [&](SceUID id) { return kernel.threads.contains(id); });
        });
        if (!gone)
            LOG_ERROR("Savestate: a thread the load ends is still there");
    }

    // The objects come back from the state, all of them
    delete_all<Callback>(kernel);
    delete_all<HeavyCond>(kernel);
    delete_all<LwCond>(kernel);
    delete_all<Semaphore>(kernel);
    delete_all<EventFlag>(kernel);
    delete_all<SimpleEvent>(kernel);
    delete_all<Timer>(kernel);
    delete_all<HeavyMutex>(kernel);
    delete_all<LwMutex>(kernel);
    delete_all<RWLock>(kernel);
    delete_all<MsgPipe>(kernel);
}

void rebuild(EmuEnvState &emuenv, const Snapshot &snapshot, const LoadPlan &plan) {
    KernelState &kernel = emuenv.kernel;
    MemState &mem = emuenv.mem;

    // No UID the restored guest holds is handed out again
    SceUID highest = 0;
    for (const ThreadRecord &r : snapshot.threads)
        highest = std::max(highest, r.id);
    for (const ObjectRecord &r : snapshot.objects)
        highest = std::max(highest, r.uid);
    while (kernel.get_next_uid() < highest) {
    }

    // Threads
    std::map<SceUID, const ThreadRecord *> records;
    for (const ThreadRecord &r : snapshot.threads)
        records.emplace(r.id, &r);
    for (const SceUID id : plan.create) {
        const ThreadRecord &r = *records.at(id);
        if (!kernel.create_thread_on(mem, r.id, r.name.c_str(), r.entry_point, r.init_priority, r.init_affinity_mask, r.stack_size, r.attr, r.stack, r.tls, r.init_cpu_ctx))
            LOG_ERROR("Savestate: thread {} \"{}\" could not be made again", r.id, r.name);
    }
    // Started, the ones the state has started go to their gate, where the suspend holds them
    for (const ThreadRecord &r : snapshot.threads) {
        if (r.resume == Resume::dormant)
            continue;
        if (const ThreadStatePtr thread = kernel.get_thread(r.id); thread && thread->get_status() == ThreadStatus::dormant)
            thread->start(0, Ptr<void>());
    }
    if (const ThreadStatePtr blocker = wait_for_all(kernel, 2000, stopped))
        LOG_ERROR("Savestate: thread {} \"{}\" did not reach its gate", blocker->id, blocker->name);

    // Objects: mutexes before the condition variables on them, callbacks once their threads exist
    const auto add = [&](std::shared_ptr<KernelObject> obj, SceUID uid) { kernel.objects.add(std::move(obj), uid); };
    const auto thread_of = [&](SceUID id) { return id ? kernel.get_thread(id) : nullptr; };
    uint32_t made = 0, missed = 0;
    std::vector<std::pair<UidClass, SceUID>> deleted_mutexes;
    const auto missing_owner = [&](const ObjectRecord &r, const std::string &name, SceUID owner) {
        LOG_WARN("Savestate: \"{}\" ({}) is held by thread {}, which the state does not have; left free", name, r.uid, owner);
    };
    for (const UidClass pass : { UidClass::mutex, UidClass::lw_mutex, UidClass::semaphore, UidClass::event_flag, UidClass::simple_event,
             UidClass::timer, UidClass::rw_lock, UidClass::msg_pipe, UidClass::cond, UidClass::lw_cond, UidClass::callback }) {
        for (const ObjectRecord &r : snapshot.objects) {
            if (r.uid_class != static_cast<uint32_t>(pass))
                continue;
            Reader in{ r.data.data(), r.data.size() };
            SceUInt32 attr = 0;
            std::string name;
            bool ok = true;
            switch (pass) {
            case UidClass::mutex:
            case UidClass::lw_mutex: {
                SceInt32 init_count = 0, count = 0;
                SceUID owner = 0;
                uint32_t workarea = 0;
                uint8_t deleted = 0;
                ok = in.get(attr) && in.get_string(name) && in.get(init_count) && (pass == UidClass::mutex || in.get(workarea)) && in.get(count) && in.get(owner)
                    && in.get(deleted);
                if (!ok)
                    break;
                std::shared_ptr<Mutex> mutex;
                if (pass == UidClass::mutex)
                    mutex = std::make_shared<HeavyMutex>(attr, name.c_str(), init_count, nullptr);
                else
                    mutex = std::make_shared<LwMutex>(attr, name.c_str(), init_count, nullptr, Ptr<SceKernelLwMutexWork>(workarea));
                add(mutex, r.uid);
                if (deleted)
                    deleted_mutexes.emplace_back(pass, r.uid);
                const ThreadStatePtr holder = thread_of(owner);
                if (owner && !holder)
                    missing_owner(r, name, owner);
                // Sets the count and the owner, with no waiters to wake. The work area is guest memory, restored already.
                mutex->cancel(holder, holder ? count : 0, nullptr);
                break;
            }
            case UidClass::semaphore: {
                SceInt32 init_val = 0, max_val = 0, count = 0;
                ok = in.get(attr) && in.get_string(name) && in.get(init_val) && in.get(max_val) && in.get(count);
                if (!ok)
                    break;
                const auto sema = std::make_shared<Semaphore>(attr, name.c_str(), init_val, max_val);
                add(sema, r.uid);
                sema->cancel(count, nullptr);
                break;
            }
            case UidClass::event_flag: {
                SceUInt32 init_pattern = 0, pattern = 0;
                ok = in.get(attr) && in.get_string(name) && in.get(init_pattern) && in.get(pattern);
                if (!ok)
                    break;
                const auto flag = std::make_shared<EventFlag>(attr, name.c_str(), init_pattern);
                add(flag, r.uid);
                flag->clear(0);
                flag->set(pattern);
                break;
            }
            case UidClass::simple_event: {
                SceUInt32 pattern = 0;
                SceUInt64 user_data = 0;
                ok = in.get(attr) && in.get_string(name) && in.get(pattern) && in.get(user_data);
                if (!ok)
                    break;
                const auto event = std::make_shared<SimpleEvent>(attr, name.c_str(), 0, 0);
                add(event, r.uid);
                event->restore(pattern, user_data);
                break;
            }
            case UidClass::timer: {
                SceInt32 active = 0, type = 0, repeat = 0;
                SceKernelSysClock time = 0, schedule = 0, interval = 0;
                SceUInt32 pattern = 0;
                SceUInt64 user_data = 0;
                ok = in.get(attr) && in.get_string(name) && in.get(active) && in.get(time) && in.get(schedule) && in.get(interval)
                    && in.get(type) && in.get(repeat) && in.get(pattern) && in.get(user_data);
                if (!ok)
                    break;
                const auto timer = std::make_shared<Timer>(attr, name.c_str(), 0);
                add(timer, r.uid);
                // Stopped, as it was made: an event set an interval before the count it is due at,
                // then the count, then the start, which queues the event for the time it has left. A
                // one shot event already past fires again here, with nobody waiting, and stays armed,
                // as it was. The pattern goes back last, over whatever that fire set.
                if (schedule != 0) {
                    timer->set_time(schedule >= interval ? schedule - interval : 0);
                    timer->set_event(type, interval, repeat);
                }
                timer->set_time(time);
                if (active)
                    timer->start();
                SceKernelTimerInfo settled{};
                timer->timer_info(settled);
                timer->restore(pattern, user_data);
                break;
            }
            case UidClass::rw_lock: {
                SceInt32 lock_count = 0;
                SceUID writer = 0;
                ok = in.get(attr) && in.get_string(name) && in.get(lock_count) && in.get(writer);
                if (!ok)
                    break;
                const auto rwlock = std::make_shared<RWLock>(attr, name.c_str());
                add(rwlock, r.uid);
                const ThreadStatePtr holder = writer ? thread_of(writer) : any_thread(kernel);
                if (writer && !holder)
                    missing_owner(r, name, writer);
                for (SceInt32 i = 0; holder && i < lock_count; i++) {
                    if (writer)
                        rwlock->try_acquire_write(holder);
                    else
                        rwlock->try_acquire_read(holder);
                }
                break;
            }
            case UidClass::msg_pipe: {
                SceInt32 buf_size = 0;
                std::vector<uint8_t> contents;
                ok = in.get(attr) && in.get_string(name) && in.get(buf_size) && in.get_bytes(contents);
                if (!ok)
                    break;
                const auto pipe = std::make_shared<MsgPipe>(attr, name.c_str(), static_cast<std::size_t>(buf_size));
                add(pipe, r.uid);
                put_back_pipe_contents(emuenv, *pipe, contents);
                break;
            }
            case UidClass::cond:
            case UidClass::lw_cond: {
                SceUID mutex_uid = 0;
                uint32_t workarea = 0;
                ok = in.get(attr) && in.get_string(name) && in.get(mutex_uid) && (pass == UidClass::cond || in.get(workarea));
                if (!ok)
                    break;
                MutexPtr mutex;
                if (pass == UidClass::cond)
                    mutex = kernel.objects.find<HeavyMutex>(mutex_uid);
                else
                    mutex = kernel.objects.find<LwMutex>(mutex_uid);
                if (!mutex) {
                    LOG_ERROR("Savestate: condition variable \"{}\" ({}) could not be made again: its mutex {} is missing", name, r.uid, mutex_uid);
                    missed++;
                    continue;
                }
                if (pass == UidClass::cond)
                    add(std::make_shared<HeavyCond>(attr, name.c_str(), std::move(mutex)), r.uid);
                else
                    add(std::make_shared<LwCond>(attr, name.c_str(), std::move(mutex), Ptr<SceKernelLwCondWork>(workarea)), r.uid);
                break;
            }
            case UidClass::callback: {
                SceUID owner_id = 0, notifier = 0;
                uint32_t func = 0, common = 0, notifications = 0;
                SceInt32 arg = 0;
                ok = in.get(owner_id) && in.get_string(name) && in.get(func) && in.get(common) && in.get(notifications) && in.get(arg) && in.get(notifier);
                if (!ok)
                    break;
                const ThreadStatePtr owner = thread_of(owner_id);
                if (!owner) {
                    LOG_ERROR("Savestate: callback \"{}\" ({}) could not be made again: its thread {} is missing", name, r.uid, owner_id);
                    missed++;
                    continue;
                }
                const auto cb = std::make_shared<Callback>(owner, name, Ptr<SceKernelCallbackFunction>(func), Ptr<void>(common));
                add(cb, r.uid);
                // The thread is held, so nothing else touches its list of callbacks
                owner->add_callback(cb);
                for (uint32_t i = 0; i < notifications; i++)
                    cb->notify(notifier, arg);
                break;
            }
            default:
                break;
            }
            if (!ok) {
                LOG_ERROR("Savestate: object {} of class {} is corrupt in the state", r.uid, r.uid_class);
                missed++;
                continue;
            }
            made++;
        }
    }

    for (const auto &[uid_class, uid] : deleted_mutexes) {
        if (uid_class == UidClass::mutex)
            kernel.objects.remove<HeavyMutex>(uid, 0);
        else
            kernel.objects.remove<LwMutex>(uid, 0);
    }

    // Registers, and what else each thread was
    uint32_t restarting = 0;
    for (const ThreadRecord &r : snapshot.threads) {
        const ThreadStatePtr thread = kernel.get_thread(r.id);
        if (!thread || r.resume == Resume::dormant)
            continue;
        thread->change_priority(r.priority);
        thread->change_affinity_mask(r.affinity_mask);
        CPUContext ctx = r.ctx;
        if (r.resume == Resume::restart_call) {
            ctx.cpu_registers[15] = ctx.get_pc() - 4;
            restarting++;
            ThreadState::ResumedWait how = ThreadState::ResumedWait::none;
            if (r.wait_type == SCE_KERNEL_WAITTYPE_COND_SIGNAL || r.wait_type == SCE_KERNEL_WAITTYPE_LW_COND_SIGNAL)
                how = ThreadState::ResumedWait::signal;
            else if (r.wait_type == SCE_KERNEL_WAITTYPE_COND_MUTEX || r.wait_type == SCE_KERNEL_WAITTYPE_LW_COND_LW_MUTEX)
                how = ThreadState::ResumedWait::relock;
            std::optional<std::chrono::microseconds> left;
            if (r.time_left_us >= 0)
                left = std::chrono::microseconds(r.time_left_us);
            thread->resume_wait(how, left);
        }
        load_context(*thread->cpu, ctx);
        write_tpidrurw(*thread->cpu, r.tpidrurw);
        while (thread->vm_suspend_count() < r.vm_suspends && thread->vm_suspend() > 0) {
        }
        while (thread->vm_suspend_count() > r.vm_suspends && thread->vm_resume() >= 0) {
        }
    }

    // The order release_threads lets the restarting threads go in
    std::vector<const ThreadRecord *> restarts;
    for (const ThreadRecord &r : snapshot.threads) {
        if (r.resume == Resume::restart_call)
            restarts.push_back(&r);
    }
    std::ranges::sort(restarts, {}, &ThreadRecord::wait_sequence);
    release_order.clear();
    for (const ThreadRecord *r : restarts)
        release_order.push_back(r->id);

    // Calls block again
    for (const ThreadStatePtr &thread : live_threads(kernel))
        thread->abandon_wait(false);
    emuenv.gxm.savestate_abandon = false;

    // Restored memory can hold other code than the JIT translated
    kernel.invalidate_jit_cache(0, 0x100000000ULL);

    LOG_INFO("Savestate: {} thread(s) kept, {} made again, {} ended; {} making their call again; {} kernel object(s) made again, {} not",
        plan.keep.size(), plan.create.size(), plan.remove.size(), restarting, made, missed);
}

} // namespace savestate::kernel
