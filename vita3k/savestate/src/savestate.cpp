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

#include <savestate/savestate.h>

#include <config/version.h>
#include <cpu/functions.h>
#include <emuenv/state.h>
#include <gxm/state.h>
#include <renderer/state.h>
#include <rtc/rtc.h>
#include <io/functions.h>
#include <io/state.h>
#include <kernel/state.h>
#include <kernel/sync_primitives.h>
#include <kernel/thread/thread_state.h>
#include <mem/functions.h>
#include <mem/state.h>
#include <gxm/functions.h>
#include <ngs/state.h>
#include <nids/functions.h>
#include <util/log.h>

#include <miniz.h>

#include <algorithm>
#include <optional>
#include <set>
#include <thread>
#include <tuple>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

namespace savestate {

namespace {

constexpr uint8_t MAGIC[8] = { 0x56, 0x49, 0x54, 0x41, 0x33, 0x4B, 0x53, 0x53 }; // "VITA3KSS"

// Chunk tags. Order in the file is not significant; the reader dispatches on the tag, and
// skips anything it does not recognise so a newer state stays partially readable.
constexpr uint32_t TAG_MEM = 0x204D454DU; // "MEM "
constexpr uint32_t TAG_CPU = 0x20555043U; // "CPU "
constexpr uint32_t TAG_WAIT = 0x54494157U; // "WAIT" - which HLE wait each parked thread is in
constexpr uint32_t TAG_SYNC = 0x434E5953U; // "SYNC" - scalar state of the kernel sync primitives
constexpr uint32_t TAG_FILE = 0x454C4946U; // "FILE" - read position of every open read-only file
// "STK " - where each thread's guest stack was. Optional, so states written before it still load.
constexpr uint32_t TAG_STACK = 0x204B5453U;
// "FTBL" - what file each open read-only descriptor refers to. Optional, so older states still load.
constexpr uint32_t TAG_FTBL = 0x4C425446U;
// "NGSL" - where each NGS system, rack and voice was. Optional, so older states still load.
constexpr uint32_t TAG_NGSL = 0x4C53474EU;
// "GXML" - where every GXM host object was. Optional, so older states still load.
constexpr uint32_t TAG_GXML = 0x4C4D5847U;
// "THRD" - each thread's status and callback depth. A load needs it to restart waits.
constexpr uint32_t TAG_THRD = 0x44524854U;
// "SYN2" - rwlock owners, message pipe contents, timers. Optional, so older states still load.
constexpr uint32_t TAG_SYN2 = 0x324E5953U;
// "GXMM" - which GPU memory regions were mapped. Optional, so older states still load.
constexpr uint32_t TAG_GXMM = 0x4D4D5847U;
// "CLCK" - how far the guest's own clock had run. Optional, so older states still load.
constexpr uint32_t TAG_CLCK = 0x4B434C43U;


// Point a context saved inside an HLE call back at the call itself, so the thread makes it again
// when it resumes. Import stubs are ARM -- `svc #0`, `mov pc, lr`, then the NID -- and a thread
// inside the call has its PC just past the svc (run_loop reads the NID at pc + 4). Its argument
// registers are untouched until the call returns, so they are still the ones it was made with.
// Checked rather than assumed; empty on success, otherwise why not.
std::string rewind_to_call(MemState &mem, CPUContext &ctx) {
    const uint32_t pc = ctx.get_pc();
    if (ctx.cpsr & 0x20)
        return fmt::format("its PC {} is in Thumb code, not an import stub", log_hex(pc));
    if (pc < 4 || !Ptr<uint32_t>(pc - 4).valid(mem) || !Ptr<uint32_t>(pc).valid(mem))
        return fmt::format("its PC {} is not in mapped memory", log_hex(pc));
    constexpr uint32_t SVC_0 = 0xEF000000U;
    constexpr uint32_t MOV_PC_LR = 0xE1A0F00EU;
    if (*Ptr<uint32_t>(pc - 4).get(mem) != SVC_0 || *Ptr<uint32_t>(pc).get(mem) != MOV_PC_LR)
        return fmt::format("its PC {} is not just past an import stub's svc", log_hex(pc));
    ctx.cpu_registers[15] = pc - 4;
    return {};
}

// A state records the emulator build that produced it. Guest memory is full of pointers into
// host-side structures whose layout this binary fixes, so a state from a different build is
// not merely stale, it is actively dangerous. Refuse it rather than crash confusingly later.
//
// app_hash is part of the identity, not just the version and build number. Those two do not
// change when the working tree is rebuilt, so without the hash the check passed for exactly the
// case it exists to catch: edit the emulator, rebuild, load a state whose host layout assumptions
// the new binary no longer shares. Anyone developing against this hits that; a released build
// never would.
std::string build_identity() {
    return std::string(app_version) + "-" + std::to_string(app_number) + "-" + app_hash;
}

template <typename T>
void put(std::vector<uint8_t> &out, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

void put_string(std::vector<uint8_t> &out, const std::string &s) {
    put<uint32_t>(out, static_cast<uint32_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}

struct Reader {
    const uint8_t *data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    bool need(size_t n) const {
        return pos + n <= size;
    }

    template <typename T>
    bool get(T &value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!need(sizeof(T)))
            return false;
        std::memcpy(&value, data + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }

    bool get_string(std::string &value) {
        uint32_t len = 0;
        if (!get(len) || !need(len))
            return false;
        value.assign(reinterpret_cast<const char *>(data + pos), len);
        pos += len;
        return true;
    }

    bool get_bytes(void *dest, size_t n) {
        if (!need(n))
            return false;
        std::memcpy(dest, data + pos, n);
        pos += n;
        return true;
    }
};

// Guest RAM is mostly zeroes and highly repetitive, so deflate takes a few hundred megabytes
// down to something tolerable to write on every quicksave.
bool deflate_to(const std::vector<uint8_t> &in, std::vector<uint8_t> &out) {
    const mz_ulong bound = mz_compressBound(static_cast<mz_ulong>(in.size()));
    out.resize(bound);
    mz_ulong produced = bound;
    const int res = mz_compress2(out.data(), &produced, in.data(),
        static_cast<mz_ulong>(in.size()), MZ_DEFAULT_COMPRESSION);
    if (res != MZ_OK)
        return false;
    out.resize(produced);
    return true;
}

bool inflate_to(const uint8_t *in, size_t in_size, std::vector<uint8_t> &out, size_t expected) {
    out.resize(expected);
    mz_ulong produced = static_cast<mz_ulong>(expected);
    const int res = mz_uncompress(out.data(), &produced, in, static_cast<mz_ulong>(in_size));
    return res == MZ_OK && produced == expected;
}

} // namespace

fs::path slot_path(const EmuEnvState &emuenv, const int slot) {
    return emuenv.vita_fs_path / "savestates" / emuenv.io.title_id / fmt::format("slot{}.vst", slot);
}

Result save(EmuEnvState &emuenv, const fs::path &path) {
    if (emuenv.io.title_id.empty())
        return Result::fail("no application is running");

    // --- is every thread one a load can restart? ------------------------------------------------
    // A load puts each thread the state has parked back at the start of its call (see "restart" in
    // load()). That is only the same as leaving it parked if the call says so -- every blocking
    // call registers how it is released and whether it can be made again (ThreadState::
    // WaitRelease) -- and only at a thread's top level: inside a callback, the host frames around
    // the call are gone. A state with a thread anywhere else would be refused by every load, so do
    // not write one. Retryable, because such moments pass.
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        for (const auto &[uid, thread] : emuenv.kernel.threads) {
            if (!thread || thread->status != ThreadStatus::wait)
                continue;
            const std::optional<ThreadState::WaitRelease> wait = thread->current_wait();
            if (!wait || !wait->restartable) {
                const uint32_t nid = thread->current_import_nid.load(std::memory_order_relaxed);
                const char *const call = nid ? import_name(nid) : nullptr;
                return Result::retry(fmt::format("thread {} \"{}\" is parked in {}, which a load could not restart",
                    uid, thread->name, call ? call : fmt::format("{}", log_hex(nid))));
            }
            if (thread->nesting_level() > 1)
                return Result::retry(fmt::format("thread {} \"{}\" is waiting inside a callback, where a load could not restart it",
                    uid, thread->name));
        }
    }

    // --- MEM: every live allocation, verbatim ---------------------------------------------
    std::vector<uint8_t> mem_raw;
    uint32_t region_count = 0;
    {
        // Pausing guest threads is not enough on its own: Vita3K's own host threads (module
        // loading, the renderer, audio) keep allocating and freeing guest memory. Without
        // holding the allocator lock across both the walk and the copy, a region can be freed
        // and decommitted between being listed and being read, and the copy then faults on
        // memory the host has already handed back.
        const std::lock_guard<std::mutex> alloc_lock(emuenv.mem.generation_mutex);

        std::vector<std::pair<Address, uint32_t>> regions;
        for_each_allocation(emuenv.mem, [&](Address addr, uint32_t size) {
            regions.emplace_back(addr, size);
        });

        size_t total = 0;
        for (const auto &region : regions)
            total += region.second;
        mem_raw.reserve(total + regions.size() * 8 + 4);

        region_count = static_cast<uint32_t>(regions.size());
        put(mem_raw, region_count);
        for (const auto &region : regions) {
            put(mem_raw, region.first);
            put(mem_raw, region.second);
            const uint8_t *src = emuenv.mem.memory.get() + region.first;
            mem_raw.insert(mem_raw.end(), src, src + region.second);
        }
    }

    std::vector<uint8_t> mem_packed;
    if (!deflate_to(mem_raw, mem_packed))
        return Result::fail("failed to compress guest memory");

    // --- CPU: one register context per live thread -----------------------------------------
    std::vector<uint8_t> cpu_raw;
    uint32_t thread_count = 0;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        std::vector<std::pair<SceUID, CPUContext>> contexts;
        for (const auto &pair : emuenv.kernel.threads) {
            if (!pair.second || !pair.second->cpu)
                continue;
            contexts.emplace_back(pair.first, save_context(*pair.second->cpu));
        }
        thread_count = static_cast<uint32_t>(contexts.size());
        put(cpu_raw, thread_count);
        for (const auto &entry : contexts) {
            put(cpu_raw, entry.first);
            put(cpu_raw, entry.second);
        }

        // A thread in ThreadStatus::wait is blocked inside a host C++ frame in one of the
        // sync primitive helpers, and WaitingThreadData holds raw pointers into that frame's
        // locals (was_canceled, outBits, result_pattern...). None of that lives in guest
        // memory or registers, so a load does not restore it: it makes the call again (see
        // "restart" in load()). Report how many there are.
        int running = 0, dormant = 0, waiting = 0, suspended = 0;
        for (const auto &pair : emuenv.kernel.threads) {
            if (!pair.second)
                continue;
            switch (pair.second->status) {
            case ThreadStatus::run: running++; break;
            case ThreadStatus::dormant: dormant++; break;
            case ThreadStatus::wait: waiting++; break;
            case ThreadStatus::suspend: suspended++; break;
            }
        }
        LOG_INFO("Savestate: thread status at snapshot - run {}, dormant {}, wait {}, suspend {}",
            running, dormant, waiting, suspended);
        LOG_INFO_IF(waiting > 0,
            "Savestate: {} thread(s) are parked inside an HLE call; a load puts each back at the start of it",
            waiting);

        // Which calls, specifically. Restoring a blocked thread means re-entering the import it
        // is parked in, so the shape of this histogram decides how big that job actually is: a
        // long tail means every wait helper needs its own restore path, while a handful of
        // repeated NIDs means a few do. Logged at save time because it is the only moment the
        // guest is quiesced with the waits still in place.
        if (waiting > 0) {
            std::map<uint32_t, int> parked_in;
            for (const auto &pair : emuenv.kernel.threads) {
                if (!pair.second || pair.second->status != ThreadStatus::wait)
                    continue;
                parked_in[pair.second->current_import_nid.load(std::memory_order_relaxed)]++;
            }
            for (const auto &[nid, count] : parked_in) {
                const char *const name = nid ? import_name(nid) : nullptr;
                LOG_INFO("Savestate:   {} thread(s) parked in {} ({})", count,
                    name ? name : "no import in flight", log_hex(nid));
            }

            // And the arguments each one was called with. A restore re-enters the call rather
            // than rebuilding its host frame, so these values are the whole input to that -- for
            // sceKernelWaitSema they are semaId, needCount and a guest pointer to the timeout.
            // Logged so the design can be checked against real values before anything relies on
            // them.
            //
            // LOG_INFO, not LOG_DEBUG: spdlog's compile-time SPDLOG_ACTIVE_LEVEL defaults to INFO,
            // so LOG_DEBUG is not merely filtered at runtime, it is compiled out. No log-level
            // setting brings it back, which makes a debug line here look like "no threads found"
            // rather than "this code cannot emit". This runs once per savestate, on user request.
            for (const auto &pair : emuenv.kernel.threads) {
                const auto &thread = pair.second;
                if (!thread || thread->status != ThreadStatus::wait)
                    continue;
                const uint32_t nid = thread->current_import_nid.load(std::memory_order_relaxed);
                const char *const name = nid ? import_name(nid) : nullptr;
                LOG_INFO("Savestate:     thread {} \"{}\" in {} args {} {} {} {}", pair.first,
                    thread->name, name ? name : "-", log_hex(thread->current_import_args[0]),
                    log_hex(thread->current_import_args[1]), log_hex(thread->current_import_args[2]),
                    log_hex(thread->current_import_args[3]));
            }
        }
    }

    // --- refuse if a frame is in flight --------------------------------------------------------
    // The display queue thread peeks the front entry, runs the guest's display callback, and only
    // then pops it -- so an empty queue means no callback is part-way through. That matters
    // because the callback's progress is tracked by renderer-side sync timestamps while the
    // SceGxmSyncObject they are compared against lives in guest memory. A load rewinds the guest
    // side and not the host side, and the queue thread is then waiting on a timestamp that will
    // never arrive: the main thread blocks in sceGxmDisplayQueueAddEntry and the guest stops.
    //
    // Snapshotting only when the queue is empty removes the inconsistency rather than trying to
    // restore around it. The caller drains it before quiescing the guest; this is the check that
    // it actually worked, and it has to be here because by now nothing else can be running.
    {
        const std::lock_guard<std::mutex> queue_lock(emuenv.gxm.display_queue.get_mutex());
        const size_t pending = emuenv.gxm.display_queue.size();
        if (pending > 0)
            return Result::retry(fmt::format("a frame is still in flight ({} display queue entries)", pending));
    }

    // Nor in the middle of drawing a scene. The renderer mirrors the guest's scene state --
    // everything the guest submitted has run by now (MainWindow::run_with_guest_quiesced drains it)
    // -- and restoring a guest outside a scene into a renderer still recording one makes the next
    // sceGxmBeginScene fail to start ("already recording") and the texture upload after it fault
    // in the driver. Measured on Disgaea 4 and Project DIVA f, about 40 ms after a load.
    if (gxm::scene_in_progress(emuenv.gxm, emuenv.mem))
        return Result::retry("a scene is still being drawn");

    // --- WAIT: what each parked thread is blocked in ------------------------------------------
    // A thread in ThreadStatus::wait is inside a host C++ frame that no snapshot can describe.
    // Rather than try to rebuild that frame on load, record what the wait *is*, so the load can
    // check the guest is still parked the same way and refuse if it is not.
    std::vector<uint8_t> wait_raw;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        std::vector<std::pair<SceUID, const ThreadState *>> parked;
        for (const auto &pair : emuenv.kernel.threads) {
            if (pair.second && pair.second->status == ThreadStatus::wait)
                parked.emplace_back(pair.first, pair.second.get());
        }
        put<uint32_t>(wait_raw, static_cast<uint32_t>(parked.size()));
        for (const auto &[uid, thread] : parked) {
            put(wait_raw, uid);
            put<uint32_t>(wait_raw, thread->current_import_nid.load(std::memory_order_relaxed));
            for (int i = 0; i < 4; i++)
                put<uint32_t>(wait_raw, thread->current_import_args[i]);
        }
    }

    // --- THRD: each thread's status and callback depth ------------------------------------------
    // A load restarts waits rather than matching them, and needs to know what a restart cannot
    // change: which threads had finished, and how deep in callbacks each one was.
    std::vector<uint8_t> thrd_raw;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        std::vector<std::tuple<SceUID, uint32_t, int32_t>> threads;
        for (const auto &[uid, thread] : emuenv.kernel.threads) {
            if (thread && thread->cpu)
                threads.emplace_back(uid, static_cast<uint32_t>(thread->status), thread->nesting_level());
        }
        put<uint32_t>(thrd_raw, static_cast<uint32_t>(threads.size()));
        for (const auto &[uid, status, level] : threads) {
            put(thrd_raw, uid);
            put(thrd_raw, status);
            put(thrd_raw, level);
        }
    }

    // --- SYNC: scalar state of the kernel sync primitives --------------------------------------
    // These objects live on the host, not in guest memory, so nothing above captures them. A
    // semaphore's count is exactly the kind of thing the guest's own bookkeeping (which *is* in
    // the snapshot) expects to agree with.
    //
    // Only the scalars are recorded, not the objects themselves: this restores into the same
    // running session, where every object still exists under the same uid. Loading a state into a
    // fresh session would need the objects constructed, which is a different and much larger job.
    std::vector<uint8_t> sync_raw;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);

        put<uint32_t>(sync_raw, static_cast<uint32_t>(emuenv.kernel.semaphores.size()));
        for (const auto &[uid, sema] : emuenv.kernel.semaphores) {
            put(sync_raw, uid);
            put<int32_t>(sync_raw, sema ? sema->val : 0);
        }

        put<uint32_t>(sync_raw, static_cast<uint32_t>(emuenv.kernel.eventflags.size()));
        for (const auto &[uid, ef] : emuenv.kernel.eventflags) {
            put(sync_raw, uid);
            put<int32_t>(sync_raw, ef ? ef->flags : 0);
        }

        put<uint32_t>(sync_raw, static_cast<uint32_t>(emuenv.kernel.simple_events.size()));
        for (const auto &[uid, ev] : emuenv.kernel.simple_events) {
            put(sync_raw, uid);
            put<uint32_t>(sync_raw, ev ? ev->pattern : 0);
            put<uint64_t>(sync_raw, ev ? ev->last_user_data : 0);
        }

        // Mutexes and lwmutexes share a type. The owner is a ThreadStatePtr; record its uid,
        // which is what a restore can look back up.
        const auto put_mutexes = [&sync_raw](const MutexPtrs &mutexes) {
            put<uint32_t>(sync_raw, static_cast<uint32_t>(mutexes.size()));
            for (const auto &[uid, mutex] : mutexes) {
                put(sync_raw, uid);
                put<int32_t>(sync_raw, mutex ? mutex->lock_count : 0);
                put<SceUID>(sync_raw, (mutex && mutex->owner) ? mutex->owner->id : 0);
            }
        };
        put_mutexes(emuenv.kernel.mutexes);
        put_mutexes(emuenv.kernel.lwmutexes);
    }

    // --- SYN2: the rest of the sync primitives ---------------------------------------------------
    // What SYNC leaves out: which threads hold each rwlock, what is sitting in each message pipe,
    // and where each timer is. Condition variables have nothing but their queue, which a load
    // rebuilds by restarting the waits. Timer deadlines are kept on a host clock that means nothing
    // to another session, so they are stored relative to now.
    std::vector<uint8_t> syn2_raw;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);

        put<uint32_t>(syn2_raw, static_cast<uint32_t>(emuenv.kernel.rwlocks.size()));
        for (const auto &[uid, rwlock] : emuenv.kernel.rwlocks) {
            put(syn2_raw, uid);
            put<uint32_t>(syn2_raw, rwlock ? static_cast<uint32_t>(rwlock->state) : 0);
            put<uint32_t>(syn2_raw, rwlock ? static_cast<uint32_t>(rwlock->owners.size()) : 0);
            if (rwlock) {
                for (const auto &[owner, count] : rwlock->owners) {
                    put<SceUID>(syn2_raw, owner ? owner->id : 0);
                    put<int32_t>(syn2_raw, count);
                }
            }
        }

        put<uint32_t>(syn2_raw, static_cast<uint32_t>(emuenv.kernel.msgpipes.size()));
        for (const auto &[uid, pipe] : emuenv.kernel.msgpipes) {
            put(syn2_raw, uid);
            std::vector<uint8_t> bytes;
            if (pipe) {
                bytes.resize(pipe->data_buffer.Used());
                pipe->data_buffer.Peek(bytes.data(), bytes.size());
            }
            put<uint32_t>(syn2_raw, static_cast<uint32_t>(bytes.size()));
            syn2_raw.insert(syn2_raw.end(), bytes.begin(), bytes.end());
        }

        const uint64_t now = sync_timer_clock();
        const auto relative = [now](uint64_t when) {
            return when == std::numeric_limits<uint64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(when - now);
        };
        put<uint32_t>(syn2_raw, static_cast<uint32_t>(emuenv.kernel.timers.size()));
        for (const auto &[uid, timer] : emuenv.kernel.timers) {
            put(syn2_raw, uid);
            const uint8_t flags = timer
                ? static_cast<uint8_t>((timer->is_started ? 1 : 0) | (timer->is_repeat ? 2 : 0) | (timer->is_pulse ? 4 : 0) | (timer->event_set ? 8 : 0))
                : 0;
            put<uint8_t>(syn2_raw, flags);
            put<uint64_t>(syn2_raw, timer ? timer->event_interval : 0);
            put<int64_t>(syn2_raw, timer ? relative(timer->next_event) : 0);
            put<int64_t>(syn2_raw, timer ? static_cast<int64_t>(timer->time - now) : 0);
        }
    }

    // --- FILE: where each open read-only file is positioned -------------------------------------
    // The offset of an open file lives in the host FILE*, not in guest memory, so nothing above
    // captures it. A load rewinds the guest's idea of how far through a file it is while the host
    // handle stays where it got to, and the two disagree from then on.
    //
    // Streamed audio is where this shows: the BGM thread reads the next chunk of a .pak into a
    // ring buffer whose indices *are* in the snapshot. Rewind those without rewinding the file and
    // the stream stops advancing -- the symptom being a fragment of music repeating.
    //
    // Read-only descriptors only. Rewinding a writable handle would leave the guest's next writes
    // landing over data it has already written -- savedata is open for writing while a save
    // dialog is up -- and that is a good deal worse than desynchronised audio. A guest that
    // reopens or re-seeks a write handle after a load recovers on its own; one whose save file
    // has been overwritten does not.
    std::vector<uint8_t> file_raw;
    uint32_t files_skipped_writable = 0;
    {
        std::vector<std::pair<SceUID, int64_t>> positions;
        for (const auto &[fd, file] : emuenv.io.std_files) {
            if (!file.is_regular_file())
                continue;
            if (file.can_write_file()) {
                files_skipped_writable++;
                continue;
            }
            const SceOff at = file.tell();
            if (at >= 0)
                positions.emplace_back(fd, static_cast<int64_t>(at));
        }
        put<uint32_t>(file_raw, static_cast<uint32_t>(positions.size()));
        for (const auto &[fd, at] : positions) {
            put(file_raw, fd);
            put(file_raw, at);
        }
        LOG_INFO("Savestate: recorded {} read-only file position(s), skipped {} writable",
            positions.size(), files_skipped_writable);
    }

    // --- what each descriptor refers to -------------------------------------------------------
    // Descriptors come from a counter that only ever goes up, so a descriptor's number depends on
    // every file the session has opened. The guest's memory holds the numbers it was given, and a
    // load rewinds that memory into a session whose counter took a different path: restoring
    // positions by number then seeks, and the guest then reads, whatever file holds that number
    // *now*. Measured as music broken after a load into a fresh session that had opened fewer
    // sound packs than the one the state was taken in -- the load reported a file "gone" and the
    // music stream read from the wrong place. Record what each number meant.
    std::vector<uint8_t> ftbl_raw;
    {
        std::vector<std::tuple<SceUID, int32_t, int64_t, std::string>> entries;
        for (const auto &[fd, file] : emuenv.io.std_files) {
            if (!file.is_regular_file() || file.can_write_file())
                continue;
            const SceOff at = file.tell();
            if (at >= 0)
                entries.emplace_back(fd, file.get_open_mode(), static_cast<int64_t>(at), std::string(file.get_vita_loc()));
        }
        put<int32_t>(ftbl_raw, static_cast<int32_t>(emuenv.io.next_fd));
        put<uint32_t>(ftbl_raw, static_cast<uint32_t>(entries.size()));
        for (const auto &[fd, flags, at, path] : entries) {
            put(ftbl_raw, fd);
            put(ftbl_raw, flags);
            put(ftbl_raw, at);
            put_string(ftbl_raw, path);
        }
    }

    // --- where each thread's stack is ---------------------------------------------------------
    // The two sessions' guest allocations diverge -- a single differently sized allocation early
    // on shifts every thread stack after it by a page -- and the load re-lays-out the address
    // space to match the state. ThreadState::stack is host-side and is not part of that, so
    // without this it keeps pointing at where the stack used to be, and the next thread to be
    // started resets its stack pointer into what is now a *different* thread's stack.
    std::vector<uint8_t> stack_raw;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        put<uint32_t>(stack_raw, static_cast<uint32_t>(emuenv.kernel.threads.size()));
        for (const auto &[id, thread] : emuenv.kernel.threads) {
            put(stack_raw, id);
            put<uint32_t>(stack_raw, thread ? thread->stack.get() : 0);
            put<int32_t>(stack_raw, thread ? thread->stack_size : 0);
        }
    }

    // --- where the NGS objects are ------------------------------------------------------------
    // See ngs::capture_layout. A load into a session whose heap put them elsewhere moves them.
    std::vector<uint8_t> ngsl_raw;
    {
        const ngs::SavedLayout layout = ngs::capture_layout(emuenv.ngs, emuenv.mem);
        put<uint32_t>(ngsl_raw, layout.definitions);
        put<uint32_t>(ngsl_raw, static_cast<uint32_t>(layout.systems.size()));
        for (const auto &system : layout.systems) {
            put<uint32_t>(ngsl_raw, system.addr);
            put<uint32_t>(ngsl_raw, static_cast<uint32_t>(system.racks.size()));
            for (const auto &rack : system.racks) {
                put<uint32_t>(ngsl_raw, rack.addr);
                put<uint32_t>(ngsl_raw, static_cast<uint32_t>(rack.voices.size()));
                for (const Address voice : rack.voices)
                    put<uint32_t>(ngsl_raw, voice);
            }
        }
    }

    // --- where the GXM objects are ------------------------------------------------------------
    // See gxm::check_layout. A load refuses a state whose objects are not all still where it had them.
    std::vector<uint8_t> gxml_raw;
    {
        const gxm::HostObjectLayout layout = gxm::capture_layout(emuenv.gxm, emuenv.mem);
        for (const std::vector<Address> *list : { &layout.contexts, &layout.render_targets, &layout.sync_objects, &layout.shader_patchers, &layout.vertex_programs, &layout.fragment_programs }) {
            put<uint32_t>(gxml_raw, static_cast<uint32_t>(list->size()));
            for (const Address address : *list)
                put<uint32_t>(gxml_raw, address);
        }
    }

    // --- which GPU memory regions are mapped ----------------------------------------------------
    // sceGxmMapMemory registrations are host state: the guest's registry, and with memory mapping
    // on a host buffer per region as well. They are not part of guest memory, so a load restores a
    // guest that expects the regions of the moment the state was taken into a session that has
    // whichever ones it has now -- and a game that maps and unmaps as it streams does not keep the
    // same set for long. A vertex stream inside a region that has since been unmapped then has no
    // buffer behind it: "Could not find matching mapped buffer for vertex stream", measured on
    // Oddworld: New 'n' Tasty right after a load, with the process going down seconds later.
    std::vector<uint8_t> gxmm_raw;
    {
        put<uint32_t>(gxmm_raw, static_cast<uint32_t>(emuenv.gxm.memory_mapped_regions.size()));
        for (const auto &[address, info] : emuenv.gxm.memory_mapped_regions) {
            put<uint32_t>(gxmm_raw, address);
            put<uint32_t>(gxmm_raw, info.size);
            put<uint32_t>(gxmm_raw, info.perm);
        }
    }

    // --- how far the guest's clock had run ------------------------------------------------------
    // sceKernelGetProcessTimeWide and friends return rtc_get_ticks(base_tick) - start_tick, which
    // is real time: the guest's clock keeps running while a state sits on disk. Nothing records it,
    // so a load leaves the guest with its own last timestamps in memory and a clock that has moved
    // on by however long ago the save was, and the game sees one enormous frame. Measured on
    // Oddworld: loading 2 s after the save resumes at 30 fps, loading 12-20 s after it resumes at
    // 14-16 fps and stays there, with the main thread burning a core in the engine's catch-up work
    // while the render worker waits for frames that never come.
    std::vector<uint8_t> clck_raw;
    put<uint64_t>(clck_raw, rtc_get_ticks(emuenv.kernel.base_tick.tick) - emuenv.kernel.start_tick);

    // --- assemble ---------------------------------------------------------------------------
    std::vector<uint8_t> file;
    file.insert(file.end(), std::begin(MAGIC), std::end(MAGIC));
    put<uint32_t>(file, FORMAT_VERSION);
    put_string(file, build_identity());
    put_string(file, emuenv.io.title_id);
    put<uint64_t>(file, static_cast<uint64_t>(std::time(nullptr)));

    put<uint32_t>(file, TAG_MEM);
    put<uint64_t>(file, static_cast<uint64_t>(mem_packed.size()));
    put<uint64_t>(file, static_cast<uint64_t>(mem_raw.size()));
    file.insert(file.end(), mem_packed.begin(), mem_packed.end());

    put<uint32_t>(file, TAG_CPU);
    put<uint64_t>(file, static_cast<uint64_t>(cpu_raw.size()));
    put<uint64_t>(file, static_cast<uint64_t>(cpu_raw.size()));
    file.insert(file.end(), cpu_raw.begin(), cpu_raw.end());

    put(file, TAG_WAIT);
    put<uint64_t>(file, wait_raw.size());
    put<uint64_t>(file, wait_raw.size());
    file.insert(file.end(), wait_raw.begin(), wait_raw.end());

    put(file, TAG_SYNC);
    put<uint64_t>(file, sync_raw.size());
    put<uint64_t>(file, sync_raw.size());
    file.insert(file.end(), sync_raw.begin(), sync_raw.end());

    put(file, TAG_FILE);
    put<uint64_t>(file, file_raw.size());
    put<uint64_t>(file, file_raw.size());
    file.insert(file.end(), file_raw.begin(), file_raw.end());

    put(file, TAG_STACK);
    put<uint64_t>(file, stack_raw.size());
    put<uint64_t>(file, stack_raw.size());
    file.insert(file.end(), stack_raw.begin(), stack_raw.end());

    put(file, TAG_FTBL);
    put<uint64_t>(file, ftbl_raw.size());
    put<uint64_t>(file, ftbl_raw.size());
    file.insert(file.end(), ftbl_raw.begin(), ftbl_raw.end());

    put(file, TAG_NGSL);
    put<uint64_t>(file, ngsl_raw.size());
    put<uint64_t>(file, ngsl_raw.size());
    file.insert(file.end(), ngsl_raw.begin(), ngsl_raw.end());

    put(file, TAG_GXML);
    put<uint64_t>(file, gxml_raw.size());
    put<uint64_t>(file, gxml_raw.size());
    file.insert(file.end(), gxml_raw.begin(), gxml_raw.end());

    put(file, TAG_THRD);
    put<uint64_t>(file, thrd_raw.size());
    put<uint64_t>(file, thrd_raw.size());
    file.insert(file.end(), thrd_raw.begin(), thrd_raw.end());

    put(file, TAG_SYN2);
    put<uint64_t>(file, syn2_raw.size());
    put<uint64_t>(file, syn2_raw.size());
    file.insert(file.end(), syn2_raw.begin(), syn2_raw.end());

    put(file, TAG_GXMM);
    put<uint64_t>(file, gxmm_raw.size());
    put<uint64_t>(file, gxmm_raw.size());
    file.insert(file.end(), gxmm_raw.begin(), gxmm_raw.end());

    put(file, TAG_CLCK);
    put<uint64_t>(file, clck_raw.size());
    put<uint64_t>(file, clck_raw.size());
    file.insert(file.end(), clck_raw.begin(), clck_raw.end());

    // Write to a temporary and rename into place, so an interrupted save cannot destroy the
    // previous good state.
    boost::system::error_code err;
    fs::create_directories(path.parent_path(), err);
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp.string(), std::ios::binary | std::ios::trunc);
        if (!out)
            return Result::fail("could not open " + tmp.string() + " for writing");
        out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
        if (!out)
            return Result::fail("write failed");
    }
    fs::remove(path, err);
    fs::rename(tmp, path, err);
    if (err)
        return Result::fail("could not move state into place: " + err.message());

    LOG_INFO("Savestate written: {} ({} regions, {} threads, {:.1f} MiB raw -> {:.1f} MiB on disk)",
        path.string(), region_count, thread_count,
        static_cast<double>(mem_raw.size()) / (1024.0 * 1024.0),
        static_cast<double>(file.size()) / (1024.0 * 1024.0));
    return Result::ok();
}


Result load(EmuEnvState &emuenv, const fs::path &path) {
    if (emuenv.io.title_id.empty())
        return Result::fail("no application is running");

    std::vector<uint8_t> file;
    {
        std::ifstream in(path.string(), std::ios::binary | std::ios::ate);
        if (!in)
            return Result::fail("no state at " + path.string());
        const auto size = static_cast<size_t>(in.tellg());
        in.seekg(0);
        file.resize(size);
        in.read(reinterpret_cast<char *>(file.data()), static_cast<std::streamsize>(size));
        if (!in)
            return Result::fail("could not read " + path.string());
    }

    Reader r{ file.data(), file.size(), 0 };

    uint8_t magic[sizeof(MAGIC)] = {};
    if (!r.get_bytes(magic, sizeof(magic)) || std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0)
        return Result::fail("not a Vita3K savestate");

    uint32_t version = 0;
    if (!r.get(version))
        return Result::fail("truncated header");
    if (version != FORMAT_VERSION)
        return Result::fail(fmt::format("state format v{}, this build reads v{}", version, FORMAT_VERSION));

    std::string build;
    if (!r.get_string(build))
        return Result::fail("truncated header");
    if (build != build_identity())
        return Result::fail(fmt::format("state came from build '{}', this is '{}'", build, build_identity()));

    std::string title_id;
    if (!r.get_string(title_id))
        return Result::fail("truncated header");
    if (title_id != emuenv.io.title_id)
        return Result::fail(fmt::format("state belongs to {}, but {} is running", title_id, emuenv.io.title_id));

    uint64_t timestamp = 0;
    if (!r.get(timestamp))
        return Result::fail("truncated header");

    // Locate every chunk before applying any of it, so a truncated file cannot leave the
    // guest half-overwritten.
    const uint8_t *mem_packed = nullptr;
    size_t mem_packed_size = 0;
    size_t mem_raw_size = 0;
    const uint8_t *cpu_data = nullptr;
    size_t cpu_size = 0;
    const uint8_t *wait_data = nullptr;
    size_t wait_size = 0;
    const uint8_t *sync_data = nullptr;
    size_t sync_size = 0;
    const uint8_t *file_data = nullptr;
    size_t file_size = 0;
    const uint8_t *stack_data = nullptr;
    size_t stack_size = 0;
    const uint8_t *ftbl_data = nullptr;
    size_t ftbl_size = 0;
    const uint8_t *ngsl_data = nullptr;
    size_t ngsl_size = 0;
    const uint8_t *gxml_data = nullptr;
    size_t gxml_size = 0;
    const uint8_t *thrd_data = nullptr;
    size_t thrd_size = 0;
    const uint8_t *syn2_data = nullptr;
    size_t syn2_size = 0;
    const uint8_t *gxmm_data = nullptr;
    size_t gxmm_size = 0;
    const uint8_t *clck_data = nullptr;
    size_t clck_size = 0;

    while (r.need(sizeof(uint32_t) + sizeof(uint64_t) * 2)) {
        uint32_t tag = 0;
        uint64_t stored = 0;
        uint64_t raw = 0;
        if (!r.get(tag) || !r.get(stored) || !r.get(raw))
            return Result::fail("truncated chunk header");
        if (!r.need(static_cast<size_t>(stored)))
            return Result::fail("truncated chunk payload");
        const uint8_t *payload = r.data + r.pos;
        r.pos += static_cast<size_t>(stored);

        switch (tag) {
        case TAG_MEM:
            mem_packed = payload;
            mem_packed_size = static_cast<size_t>(stored);
            mem_raw_size = static_cast<size_t>(raw);
            break;
        case TAG_CPU:
            cpu_data = payload;
            cpu_size = static_cast<size_t>(stored);
            break;
        case TAG_WAIT:
            wait_data = payload;
            wait_size = static_cast<size_t>(stored);
            break;
        case TAG_SYNC:
            sync_data = payload;
            sync_size = static_cast<size_t>(stored);
            break;
        case TAG_FILE:
            file_data = payload;
            file_size = static_cast<size_t>(stored);
            break;
        case TAG_STACK:
            stack_data = payload;
            stack_size = static_cast<size_t>(stored);
            break;
        case TAG_FTBL:
            ftbl_data = payload;
            ftbl_size = static_cast<size_t>(stored);
            break;
        case TAG_NGSL:
            ngsl_data = payload;
            ngsl_size = static_cast<size_t>(stored);
            break;
        case TAG_GXML:
            gxml_data = payload;
            gxml_size = static_cast<size_t>(stored);
            break;
        case TAG_THRD:
            thrd_data = payload;
            thrd_size = static_cast<size_t>(stored);
            break;
        case TAG_SYN2:
            syn2_data = payload;
            syn2_size = static_cast<size_t>(stored);
            break;
        case TAG_GXMM:
            gxmm_data = payload;
            gxmm_size = static_cast<size_t>(stored);
            break;
        case TAG_CLCK:
            clck_data = payload;
            clck_size = static_cast<size_t>(stored);
            break;
        default:
            LOG_WARN("Savestate: ignoring unknown chunk 0x{:08X}", tag);
            break;
        }
    }

    if (!mem_packed || !cpu_data || !wait_data || !sync_data || !file_data)
        return Result::fail("state is missing a required chunk");
    if (!thrd_data)
        return Result::fail("state predates restarting waits on load; take a new one");

    // The display queue must be empty here, for the same reason the save requires it, and it has
    // to be re-checked now rather than trusted from before the pause. The queue's host thread
    // drives a *guest* thread: it calls run_guest_function on SceGxmDisplayQueue, which resets
    // that thread's program counter and stack pointer and sets it running. Restoring thread
    // contexts while that is in flight leaves the thread resuming from a stale program counter --
    // measured as a write to 0x400000000 from JIT code on "guest 337 SceGxmDisplayQueue", after
    // which the frame loop stopped advancing and the last frame was presented forever.
    //
    // With the queue empty the host thread is blocked waiting for an entry and the guest that
    // would push one is paused, so no callback can start. Retryable: the boundary comes round
    // every frame.
    {
        const std::lock_guard<std::mutex> queue_lock(emuenv.gxm.display_queue.get_mutex());
        const size_t pending = emuenv.gxm.display_queue.size();
        if (pending > 0)
            return Result::retry(fmt::format("a frame is still in flight ({} display queue entries)", pending));
    }

    // Nor in the middle of drawing a scene. The renderer mirrors the guest's scene state --
    // everything the guest submitted has run by now (MainWindow::run_with_guest_quiesced drains it)
    // -- and restoring a guest outside a scene into a renderer still recording one makes the next
    // sceGxmBeginScene fail to start ("already recording") and the texture upload after it fault
    // in the driver. Measured on Disgaea 4 and Project DIVA f, about 40 ms after a load.
    if (gxm::scene_in_progress(emuenv.gxm, emuenv.mem))
        return Result::retry("a scene is still being drawn");

    // --- which threads the load has to restart --------------------------------------------------
    // A thread parked in an HLE call is inside a host C++ frame that no snapshot describes. Rather
    // than require this session's threads to be parked exactly as the state's were, every thread
    // waiting now is taken out of its call, and every thread the state had parked is put back at
    // the start of its call so that it makes it again ("restart", below). What still has to agree
    // is what a restart cannot change: which threads exist, which have finished, and how deep in
    // callbacks each one is.
    struct ParkedWait {
        uint32_t nid;
        uint32_t args[4];
    };
    std::map<SceUID, ParkedWait> parked;
    {
        Reader wr{ wait_data, wait_size, 0 };
        uint32_t count = 0;
        if (!wr.get(count))
            return Result::fail("corrupt wait chunk");
        for (uint32_t i = 0; i < count; i++) {
            SceUID id = 0;
            ParkedWait w{};
            if (!wr.get(id) || !wr.get(w.nid))
                return Result::fail("corrupt wait chunk");
            for (uint32_t &arg : w.args) {
                if (!wr.get(arg))
                    return Result::fail("corrupt wait chunk");
            }
            parked[id] = w;
        }
    }

    struct SavedThread {
        ThreadStatus status;
        int32_t level;
    };
    std::map<SceUID, SavedThread> saved_threads;
    {
        Reader tr{ thrd_data, thrd_size, 0 };
        uint32_t count = 0;
        if (!tr.get(count))
            return Result::fail("corrupt thread chunk");
        for (uint32_t i = 0; i < count; i++) {
            SceUID id = 0;
            uint32_t status = 0;
            int32_t level = 0;
            if (!tr.get(id) || !tr.get(status) || !tr.get(level))
                return Result::fail("corrupt thread chunk");
            saved_threads[id] = { static_cast<ThreadStatus>(status), level };
        }
    }

    std::map<SceUID, CPUContext> contexts;
    {
        Reader cr{ cpu_data, cpu_size, 0 };
        uint32_t count = 0;
        if (!cr.get(count))
            return Result::fail("corrupt cpu chunk");
        for (uint32_t i = 0; i < count; i++) {
            SceUID id = 0;
            CPUContext ctx;
            if (!cr.get(id) || !cr.get(ctx))
                return Result::fail("corrupt cpu chunk");
            contexts[id] = ctx;
        }
    }

    // A thread waiting now, and how to end its call.
    struct Release {
        ThreadStatePtr thread;
        uint32_t nid;
        std::function<void()> release; // empty: ends on its own
    };
    std::vector<Release> releases;
    uint32_t restarts = 0;
    uint32_t differed = 0;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        for (auto &[id, ctx] : contexts) {
            const auto it = emuenv.kernel.threads.find(id);
            if (it == emuenv.kernel.threads.end() || !it->second || !it->second->cpu)
                return Result::fail(fmt::format("thread {} from the state no longer exists", id));
            const ThreadStatePtr &thread = it->second;
            const auto saved = saved_threads.find(id);
            if (saved == saved_threads.end())
                return Result::fail(fmt::format("the state has registers for thread {} but no status", id));

            // What a restart cannot change has to agree already. Retryable: a thread that has just
            // finished, or is in a callback, is usually back where it was a frame or two later.
            const bool finished_now = thread->status == ThreadStatus::dormant;
            const bool finished_then = saved->second.status == ThreadStatus::dormant;
            if (finished_now != finished_then)
                return Result::retry(fmt::format("thread {} \"{}\" {} now but {} when the state was taken", id, thread->name,
                    finished_now ? "has finished" : "is running", finished_then ? "had finished" : "was running"));
            if (thread->nesting_level() != saved->second.level)
                return Result::retry(fmt::format("thread {} \"{}\" is {} callback(s) deep now but was {} when the state was taken",
                    id, thread->name, thread->nesting_level() - 1, saved->second.level - 1));

            // The state's side: a thread parked in a call goes back to the start of it.
            const auto p = parked.find(id);
            if (p != parked.end()) {
                // The save only writes a state whose parked threads are all in calls that can be made
                // again (see the check at the top of save()).
                if (p->second.nid == 0)
                    return Result::fail(fmt::format("thread {} was parked outside any call", id));
                if (saved->second.level != 1)
                    return Result::fail(fmt::format("thread {} was parked inside a callback, where a load cannot restart it", id));
                const std::string why = rewind_to_call(emuenv.mem, ctx);
                if (!why.empty())
                    return Result::fail(fmt::format("thread {} cannot be restarted: {}", id, why));
                restarts++;
            }

            // This session's side: a thread waiting now is taken out of its call.
            if (thread->status == ThreadStatus::wait) {
                const uint32_t nid = thread->current_import_nid.load(std::memory_order_relaxed);
                const char *const name = nid ? import_name(nid) : nullptr;
                const std::string call = name ? name : fmt::format("{}", log_hex(nid));
                if (thread->nesting_level() != 1)
                    return Result::retry(fmt::format("thread {} \"{}\" is in {} inside a callback", id, thread->name, call));
                const std::optional<ThreadState::WaitRelease> wait = thread->current_wait();
                if (!wait)
                    return Result::retry(fmt::format("thread {} \"{}\" is in {}, which a load cannot interrupt", id, thread->name, call));
                releases.push_back({ thread, nid, wait->release });
            }

            // For the log only: how many threads the all-or-nothing match this replaced would have
            // refused the load over.
            const bool parked_then = p != parked.end();
            const bool parked_now = thread->status == ThreadStatus::wait;
            if (parked_then != parked_now
                || (parked_now
                    && (p->second.nid != thread->current_import_nid.load(std::memory_order_relaxed)
                        || !std::equal(p->second.args, p->second.args + 3, thread->current_import_args))))
                differed++;
        }
    }

    // --- are the GXM objects the state refers to still where it had them? ---------------------
    // They are not moved by a load, so if the game's heap put them elsewhere in this session the
    // restored guest would hand back addresses with no live object behind them. Refuse instead.
    std::optional<gxm::HostObjectLayout> gxm_layout;
    if (gxml_data) {
        Reader gr{ gxml_data, gxml_size, 0 };
        gxm::HostObjectLayout layout;
        for (std::vector<Address> *list : { &layout.contexts, &layout.render_targets, &layout.sync_objects, &layout.shader_patchers, &layout.vertex_programs, &layout.fragment_programs }) {
            uint32_t count = 0;
            if (!gr.get(count))
                return Result::fail("corrupt GXM layout chunk");
            list->resize(count);
            for (Address &address : *list) {
                if (!gr.get(address))
                    return Result::fail("corrupt GXM layout chunk");
            }
        }
        const std::string why = gxm::check_layout(emuenv.gxm, emuenv.mem, layout);
        if (!why.empty())
            return Result::fail("the graphics objects are laid out differently in this session: " + why);
        gxm_layout = std::move(layout);
    }

    // --- can the NGS objects be moved to where the state had them? ----------------------------
    // Decided here, before anything is touched, so a refusal leaves the session as it was.
    std::optional<ngs::SavedLayout> ngs_layout;
    if (ngsl_data) {
        Reader nr{ ngsl_data, ngsl_size, 0 };
        ngs::SavedLayout layout;
        uint32_t system_count = 0;
        if (!nr.get(layout.definitions) || !nr.get(system_count))
            return Result::fail("corrupt NGS layout chunk");
        for (uint32_t i = 0; i < system_count; i++) {
            ngs::SavedLayout::SystemEntry system;
            uint32_t rack_count = 0;
            if (!nr.get(system.addr) || !nr.get(rack_count))
                return Result::fail("corrupt NGS layout chunk");
            for (uint32_t j = 0; j < rack_count; j++) {
                ngs::SavedLayout::RackEntry rack;
                uint32_t voice_count = 0;
                if (!nr.get(rack.addr) || !nr.get(voice_count))
                    return Result::fail("corrupt NGS layout chunk");
                rack.voices.resize(voice_count);
                for (Address &voice : rack.voices) {
                    if (!nr.get(voice))
                        return Result::fail("corrupt NGS layout chunk");
                }
                system.racks.push_back(std::move(rack));
            }
            layout.systems.push_back(std::move(system));
        }
        if (ngs::scheduler_busy(emuenv.ngs))
            return Result::retry("the NGS scheduler is busy");
        const std::string why = ngs::check_relocatable(emuenv.ngs, emuenv.mem, layout);
        if (!why.empty())
            return Result::fail("the audio engine cannot be moved to where the state has it: " + why);
        ngs_layout = std::move(layout);
    }

    std::vector<uint8_t> mem_raw;
    if (!inflate_to(mem_packed, mem_packed_size, mem_raw, mem_raw_size))
        return Result::fail("guest memory chunk is corrupt");

    // --- put the GPU memory mappings back to what the state had -------------------------------
    // In two halves, because unmapping and mapping have to sit on either side of the memory
    // restore. With the page table in use, remove_external_mapping copies the host buffer back
    // over the guest's own pages as it tears a mapping down, so a mapping dropped *after* the
    // restore would write stale bytes over restored ones. Mapping has the mirror requirement: it
    // copies the guest's pages into the new buffer, so it has to see the restored contents.
    //
    // Unmap first, then. Anything mapped now that the state did not have at the same address and
    // size goes; the rest is left alone, because tearing a mapping down and building it again
    // costs a buffer allocation and a copy of the whole region for nothing.
    std::vector<MemoryMapInfo> state_mappings;
    if (gxmm_data) {
        Reader gr{ gxmm_data, gxmm_size, 0 };
        uint32_t count = 0;
        if (!gr.get(count))
            return Result::fail("corrupt GPU memory mapping chunk");
        state_mappings.resize(count);
        for (MemoryMapInfo &info : state_mappings) {
            if (!gr.get(info.offset) || !gr.get(info.size) || !gr.get(info.perm))
                return Result::fail("corrupt GPU memory mapping chunk");
        }
    }

    size_t mappings_dropped = 0;
    size_t mappings_added = 0;
    if (gxmm_data) {
        const auto state_has = [&](Address address, uint32_t size) {
            return std::any_of(state_mappings.begin(), state_mappings.end(),
                [&](const MemoryMapInfo &info) { return info.offset == address && info.size == size; });
        };
        for (auto ite = emuenv.gxm.memory_mapped_regions.begin(); ite != emuenv.gxm.memory_mapped_regions.end();) {
            if (state_has(ite->first, ite->second.size)) {
                ++ite;
                continue;
            }
            if (emuenv.renderer->features.enable_memory_mapping && ite->second.size > 0)
                emuenv.renderer->unmap_memory(emuenv.mem, Ptr<void>(ite->first));
            ite = emuenv.gxm.memory_mapped_regions.erase(ite);
            mappings_dropped++;
        }
    }

    // --- reconcile the address space --------------------------------------------------------
    // Restoring page contents is only meaningful if the same pages are allocated. Allocations
    // made since the save must go, and allocations freed since must come back, or the guest
    // resumes against an address space that does not match its own pointers. Doing this
    // through the allocator rather than by restoring the allocation table directly keeps the
    // host commit state correct: alloc/free commit and decommit pages as they go.
    Reader mr{ mem_raw.data(), mem_raw.size(), 0 };
    uint32_t region_count = 0;
    if (!mr.get(region_count))
        return Result::fail("corrupt memory chunk");

    struct SavedRegion {
        Address addr;
        uint32_t size;
        size_t offset; // into mem_raw, where this region's bytes start
    };
    std::vector<SavedRegion> saved;
    saved.reserve(region_count);
    for (uint32_t i = 0; i < region_count; i++) {
        Address addr = 0;
        uint32_t size = 0;
        if (!mr.get(addr) || !mr.get(size) || !mr.need(size))
            return Result::fail("corrupt memory chunk");
        saved.push_back({ addr, size, mr.pos });
        mr.pos += size;
    }

    // Reconcile the address space, then restore, all under one hold of generation_mutex.
    //
    // The regions the state does not have are released with free_for_savestate, which gives their
    // pages back to the allocator WITHOUT decommitting them. That distinction is the whole fix.
    // Host-side structures outside the guest -- the renderer, the texture cache, the GXM display
    // queue -- hold raw pointers into guest memory and are never told a region has gone. When the
    // ordinary free() decommitted during a load, the next one to touch an unmapped page faulted:
    // measured repeatedly as a crash in memmove against an unmapped host address, with null
    // dereferences on other threads immediately behind it. Leaving the pages mapped turns that
    // into a stale read, and the restore below overwrites the bytes for every region the state
    // actually has.
    //
    // Locking the whole operation was tried first, on the theory that the window between free and
    // realloc was the problem. It is not -- the faults are host code following pointers that were
    // already stale -- but the single hold is kept, because the walk and the copy do want to be
    // atomic with respect to the guest.
    // Regions that hold host objects placement-new'd into guest memory. Their bytes are captured
    // like any other guest memory, but they contain this-process host pointers, so restoring a
    // state taken by a different process installs pointers into a heap that no longer exists.
    // Collected before anything is overwritten, while the walk is still safe.
    // First, the objects the guest created after the state was taken. They are live here and the
    // state has no bytes for them, but their guest memory is about to be put back to a moment when
    // it was free, and the guest then hands it out again while the host still caches the object.
    // See gxm::rollback_to_layout for the measurement. Done before the ranges are collected, so
    // theirs are not stepped over: the restore writes what the guest had there. The one refusal
    // this can raise happens before anything is touched.
    if (gxm_layout) {
        gxm::RollbackCounts rolled;
        const std::string why = gxm::rollback_to_layout(emuenv.gxm, emuenv.mem, *gxm_layout, rolled);
        if (!why.empty())
            return Result::fail("the graphics objects cannot be put back to what the state had: " + why);
        if (rolled.total() > 0)
            LOG_INFO("Savestate: destroyed {} vertex program(s), {} fragment program(s), {} shader patcher(s) and {} sync object(s) the guest created after the save",
                rolled.vertex_programs, rolled.fragment_programs, rolled.shader_patchers, rolled.sync_objects);
    }
    std::vector<std::pair<Address, uint32_t>> host_owned_ranges;
    gxm::collect_host_owned_ranges(emuenv.gxm, emuenv.mem, host_owned_ranges);

    // NGS objects. With a recorded layout, copy the live objects aside now -- before anything is
    // overwritten or reallocated -- and keep the restore off the addresses the state has them at,
    // where they are placed afterwards. Without one (older states), keep them where they are.
    std::optional<ngs::RelocationStash> ngs_stash;
    if (ngs_layout) {
        ngs_stash = ngs::stash_for_relocation(emuenv.ngs, emuenv.mem, *ngs_layout);
        const size_t graphics_ranges = host_owned_ranges.size();
        for (const auto &block : ngs_stash->blocks) {
            const uint32_t size = static_cast<uint32_t>(block.bytes.size());
            for (size_t i = 0; i < graphics_ranges; i++) {
                const auto &[addr, length] = host_owned_ranges[i];
                if (block.to < addr + length && addr < block.to + size)
                    return Result::fail(fmt::format("the audio engine's saved location {} collides with a graphics object here", log_hex(block.to)));
            }
            host_owned_ranges.emplace_back(block.to, size);
        }
    } else {
        ngs::collect_host_owned_ranges(emuenv.ngs, emuenv.mem, host_owned_ranges);
    }

    // --- restart: take every waiting thread out of its call ------------------------------------
    // Nothing from here on refuses, so this is the point of no return. Each thread waiting now is
    // armed to continue from the state's registers once its call returns
    // (ThreadState::continue_from_on_return), and its call is then ended through the release its
    // wait registered (ThreadState::WaitRelease); the thread parks on savestate_lock until this load
    // lets go of it. A wait with no release -- sceAudioOutOutput, blocked on that same lock --
    // cannot be woken and does not need to be: it reaches its armed registers once the load ends.
    std::set<SceUID> armed;
    std::vector<ThreadStatePtr> left_on_their_own;
    for (const Release &r : releases) {
        if (!r.thread->continue_from_on_return(r.nid, contexts.at(r.thread->id))) {
            // Its call ended between the check above and here -- a delay ran out -- and the pause
            // parks it on the way out (KernelState::pause_threads). It is restored below like any
            // other stopped thread, once it has actually stopped.
            left_on_their_own.push_back(r.thread);
            continue;
        }
        armed.insert(r.thread->id);
        // Does nothing if something ended the wait in the meantime, which ends the call anyway.
        if (r.release)
            r.release();
    }

    // Let the woken ones get out of their calls before memory is touched: on the way out they still
    // write to it (sceKernelWaitSema writes back the time left).
    const auto wait_until = [](const ThreadStatePtr &thread, const auto &done) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (!done() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        if (!done())
            LOG_ERROR("Savestate: thread {} \"{}\" did not get out of its call; restoring anyway", thread->id, thread->name);
    };
    for (const Release &r : releases) {
        if (armed.contains(r.thread->id) && r.release)
            wait_until(r.thread, [&] { return r.thread->held_for_savestate(); });
    }
    for (const ThreadStatePtr &thread : left_on_their_own) {
        wait_until(thread, [&] {
            const std::lock_guard<std::mutex> thread_lock(thread->mutex);
            return thread->status == ThreadStatus::suspend;
        });
    }

    // Before anything touches guest memory. The renderer holds read-only protections over the
    // textures and surfaces it has cached; hand them all back, and let their callbacks invalidate
    // those caches, rather than tripping a page fault per range in the middle of the restore.
    // Taken before generation_mutex so the two are never nested.
    drop_all_protections(emuenv.mem);

    uint32_t freed = 0, reallocated = 0, unrecoverable = 0, skipped = 0;
    uint32_t host_owned_ranges_skipped = 0;
    uint32_t left_alone = 0;
    {
        const std::lock_guard<std::mutex> alloc_lock(emuenv.mem.generation_mutex);

        std::map<Address, uint32_t> live;
        for_each_allocation(emuenv.mem, [&](Address addr, uint32_t size) {
            live.emplace(addr, size);
        });

        std::map<Address, uint32_t> wanted;
        for (const auto &region : saved)
            wanted.emplace(region.addr, region.size);

        // Release as little as possible.
        //
        // Every allocation the guest has made since the state was taken is also referenced by
        // something host-side that the load does not know about -- a queued display callback's
        // data, a renderer buffer, a texture. Releasing one gives its pages back to the allocator
        // while those references still exist, and the session dies about two seconds later:
        // measured repeatedly, and only ever on a load that released something. A load that
        // released nothing always ran.
        //
        // Leaving an extra allocation alone costs a few pages that the guest will never ask about
        // -- the allocator is host-side, so the restored guest cannot see it, and it will not be
        // handed out again. So free only where it is actually necessary: a region at the same
        // address with a different size, or one standing on ground the state needs.
        for (const auto &entry : live) {
            const auto it = wanted.find(entry.first);
            if (it != wanted.end() && it->second == entry.second)
                continue;

            const bool wrong_size = (it != wanted.end());
            const bool in_the_way = std::any_of(wanted.begin(), wanted.end(), [&](const auto &want) {
                if (live.find(want.first) != live.end() && live.at(want.first) == want.second)
                    return false; // already there, will not be allocated
                return want.first < entry.first + entry.second && entry.first < want.first + want.second;
            });

            if (!wrong_size && !in_the_way) {
                left_alone++;
                continue;
            }

            LOG_INFO("Savestate: releasing region at {} ({} bytes, state {}, {})",
                log_hex(entry.first), entry.second,
                wrong_size ? fmt::format("has {} bytes", it->second) : "does not have it",
                in_the_way ? "in the way of one the state needs" : "wrong size");
            free_for_savestate(emuenv.mem, entry.first);
            freed++;
        }

        for (const auto &entry : wanted) {
            const auto it = live.find(entry.first);
            if (it != live.end() && it->second == entry.second)
                continue;
            LOG_INFO("Savestate: restoring region at {} ({} bytes)", log_hex(entry.first), entry.second);
            if (try_alloc_at_locked(emuenv.mem, entry.first, entry.second, "savestate") == 0) {
                unrecoverable++;
                continue;
            }
            reallocated++;
        }

        for (const auto &region : saved) {
            if (!Ptr<uint8_t>(region.addr).valid(emuenv.mem)) {
                skipped++;
                continue;
            }
            // Copy the region, stepping over any host-owned range inside it. The ranges are few
            // and small, so a linear scan per region is cheaper than any indexing would be.
            uint8_t *const dest = emuenv.mem.memory.get() + region.addr;
            const uint8_t *const src = mem_raw.data() + region.offset;
            uint32_t at = 0;
            while (at < region.size) {
                uint32_t next_hole_start = region.size;
                uint32_t next_hole_end = region.size;
                for (const auto &[hole_addr, hole_size] : host_owned_ranges) {
                    if (hole_addr + hole_size <= region.addr || hole_addr >= region.addr + region.size)
                        continue;
                    const uint32_t start = (hole_addr <= region.addr) ? 0 : hole_addr - region.addr;
                    const uint32_t end = std::min(region.size, (hole_addr + hole_size) - region.addr);
                    if (end <= at)
                        continue;
                    if (start < next_hole_start) {
                        next_hole_start = std::max(start, at);
                        next_hole_end = end;
                    }
                }

                if (next_hole_start > at)
                    std::memcpy(dest + at, src + at, next_hole_start - at);
                if (next_hole_end > next_hole_start)
                    host_owned_ranges_skipped++;
                at = std::max(next_hole_end, next_hole_start);
                if (next_hole_start == region.size)
                    break;
            }
        }
    }

    // The second half: every region the state had that is not mapped now. After the restore, so
    // that a mapping which copies the guest's pages into its buffer copies the restored ones.
    // Called straight rather than through CommandOpcode::MemoryMap, because the render thread is
    // parked for the whole load and would never pick the command up.
    for (const MemoryMapInfo &info : state_mappings) {
        if (emuenv.gxm.memory_mapped_regions.contains(info.offset))
            continue;
        emuenv.gxm.memory_mapped_regions.emplace(info.offset, info);
        if (emuenv.renderer->features.enable_memory_mapping && info.size > 0)
            emuenv.renderer->map_memory(emuenv.mem, Ptr<void>(info.offset), info.size);
        mappings_added++;
    }
    if (mappings_dropped > 0 || mappings_added > 0)
        LOG_INFO("Savestate: GPU memory mappings reconciled ({} dropped, {} restored, {} left alone)",
            mappings_dropped, mappings_added, emuenv.gxm.memory_mapped_regions.size() - mappings_added);

    if (freed > 0 || reallocated > 0 || left_alone > 0)
        LOG_INFO("Savestate: address space reconciled ({} released, {} restored, {} left alone)",
            freed, reallocated, left_alone);
    if (unrecoverable > 0)
        LOG_ERROR("Savestate: {} region(s) could not be re-allocated; the guest address space does not match the state", unrecoverable);

    if (skipped > 0)
        LOG_WARN("Savestate: {} region(s) are still not mapped and were skipped", skipped);

    if (host_owned_ranges_skipped > 0)
        LOG_INFO("Savestate: stepped over {} host-owned range(s) inside restored regions", host_owned_ranges_skipped);

    if (ngs_stash) {
        const uint32_t moved = ngs::relocate_after_restore(emuenv.ngs, emuenv.mem, *ngs_stash);
        LOG_INFO_IF(moved > 0, "Savestate: moved {} NGS pool(s) to where the state has them", moved);
    }

    // --- point each thread at its stack -------------------------------------------------------
    // Must happen after the address space has been reconciled and before any thread runs again.
    // Optional: states written before this chunk existed simply do not carry it, and are loaded
    // as they were.
    if (stack_data) {
        Reader sk{ stack_data, stack_size, 0 };
        uint32_t count = 0;
        if (!sk.get(count))
            return Result::fail("corrupt stack chunk");

        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        uint32_t moved = 0;
        for (uint32_t i = 0; i < count; i++) {
            SceUID id = 0;
            uint32_t addr = 0;
            int32_t size = 0;
            if (!sk.get(id) || !sk.get(addr) || !sk.get(size))
                return Result::fail("corrupt stack chunk");

            const auto it = emuenv.kernel.threads.find(id);
            if (it == emuenv.kernel.threads.end() || !it->second || addr == 0)
                continue;
            const auto &thread = it->second;
            if (thread->stack.get() == addr)
                continue;

            // The old allocation was released by the reconciliation above, so the Block must not
            // free it again.
            thread->stack.rebase_for_savestate(addr);
            thread->stack_size = size;
            moved++;
        }

        LOG_INFO_IF(moved > 0, "Savestate: {} thread stack(s) moved to where the state has them", moved);
    }

    // --- restore CPU contexts -----------------------------------------------------------------
    // Threads armed above pick theirs up when their call returns. Every other thread is stopped,
    // and gets it now.
    uint32_t restored = 0;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        for (const auto &[id, ctx] : contexts) {
            const auto it = emuenv.kernel.threads.find(id);
            if (it == emuenv.kernel.threads.end() || !it->second || !it->second->cpu)
                continue;
            if (!armed.contains(id))
                load_context(*it->second->cpu, ctx);
            restored++;
        }

        // Restoring memory can change code the JIT has already translated, so every cached
        // block has to go. Without this the CPU happily keeps executing translations of
        // instructions that no longer exist.
        for (const auto &pair : emuenv.kernel.threads) {
            if (pair.second && pair.second->cpu)
                invalidate_jit_cache(*pair.second->cpu, 0, std::numeric_limits<uint32_t>::max());
        }
    }
    const uint32_t thread_count = static_cast<uint32_t>(contexts.size());
    LOG_INFO("Savestate: {} thread(s) taken out of their calls, {} put back at the start of theirs ({} of them parked differently than the state had them)",
        armed.size(), restarts, differed);

    // --- restore sync primitive scalars --------------------------------------------------------
    // The guest's own bookkeeping is in the snapshot; these host-side counters are what it expects
    // to agree with. Objects that no longer exist are skipped rather than created: the gate above
    // has already established that the threads which matter are parked exactly as they were, and
    // creating kernel objects here would be reconstructing state this design deliberately does not
    // own.
    uint32_t sync_restored = 0, sync_skipped = 0;
    {
        Reader sr{ sync_data, sync_size, 0 };
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);

        const auto read_table = [&](const auto &table, auto &&apply) -> bool {
            uint32_t count = 0;
            if (!sr.get(count))
                return false;
            for (uint32_t i = 0; i < count; i++) {
                SceUID uid = 0;
                if (!sr.get(uid))
                    return false;
                const auto it = table.find(uid);
                if (!apply(it != table.end() ? it->second : nullptr))
                    return false;
            }
            return true;
        };

        const auto count_it = [&](bool present) { present ? sync_restored++ : sync_skipped++; };

        bool ok = read_table(emuenv.kernel.semaphores, [&](const SemaphorePtr &sema) {
            int32_t val = 0;
            if (!sr.get(val))
                return false;
            if (sema)
                sema->val = val;
            count_it(sema != nullptr);
            return true;
        });

        ok = ok && read_table(emuenv.kernel.eventflags, [&](const EventFlagPtr &ef) {
            int32_t flags = 0;
            if (!sr.get(flags))
                return false;
            if (ef)
                ef->flags = flags;
            count_it(ef != nullptr);
            return true;
        });

        ok = ok && read_table(emuenv.kernel.simple_events, [&](const SimpleEventPtr &ev) {
            uint32_t pattern = 0;
            uint64_t user_data = 0;
            if (!sr.get(pattern) || !sr.get(user_data))
                return false;
            if (ev) {
                ev->pattern = pattern;
                ev->last_user_data = user_data;
            }
            count_it(ev != nullptr);
            return true;
        });

        const auto restore_mutexes = [&](const MutexPtrs &table) {
            return read_table(table, [&](const MutexPtr &mutex) {
                int32_t lock_count = 0;
                SceUID owner_id = 0;
                if (!sr.get(lock_count) || !sr.get(owner_id))
                    return false;
                if (mutex) {
                    mutex->lock_count = lock_count;
                    const auto owner = emuenv.kernel.threads.find(owner_id);
                    mutex->owner = (owner != emuenv.kernel.threads.end()) ? owner->second : nullptr;
                }
                count_it(mutex != nullptr);
                return true;
            });
        };
        ok = ok && restore_mutexes(emuenv.kernel.mutexes);
        ok = ok && restore_mutexes(emuenv.kernel.lwmutexes);

        if (!ok)
            return Result::fail("corrupt sync chunk");
    }

    // --- put the guest's clock back where the state left it ---------------------------------------
    // Without this the guest resumes with its own timestamps in memory and a clock that has run on
    // by however long the state sat unused, so the first frame after a load is as long as that gap.
    // start_tick is the origin sceKernelGetProcessTimeWide measures from, so moving it forward by
    // the gap makes the guest's clock continue rather than jump. The wall clock is left alone:
    // sceRtcGetCurrentTick really should say what time it is now.
    if (clck_data) {
        Reader cr{ clck_data, clck_size, 0 };
        uint64_t guest_elapsed = 0;
        if (!cr.get(guest_elapsed)) {
            LOG_ERROR("Savestate: corrupt clock chunk; the guest's clock will jump");
        } else {
            const uint64_t now = rtc_get_ticks(emuenv.kernel.base_tick.tick);
            const uint64_t was = now - emuenv.kernel.start_tick;
            emuenv.kernel.start_tick = now - guest_elapsed;
            LOG_INFO("Savestate: guest clock rewound {} us, to {} us", was - guest_elapsed, guest_elapsed);
        }
    }

    // --- restore the rest of the sync primitives (SYN2) ------------------------------------------
    // Past the point of no return, so a chunk that does not parse is reported, not refused.
    if (syn2_data) {
        Reader xr{ syn2_data, syn2_size, 0 };
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        uint32_t count = 0;
        bool ok = xr.get(count);
        for (uint32_t i = 0; ok && i < count; i++) {
            SceUID uid = 0;
            uint32_t state = 0, owner_count = 0;
            ok = xr.get(uid) && xr.get(state) && xr.get(owner_count);
            RWLockOwners owners;
            for (uint32_t j = 0; ok && j < owner_count; j++) {
                SceUID owner = 0;
                int32_t held = 0;
                ok = xr.get(owner) && xr.get(held);
                const auto t = emuenv.kernel.threads.find(owner);
                if (ok && t != emuenv.kernel.threads.end() && t->second)
                    owners.emplace(t->second, held);
            }
            const auto it = emuenv.kernel.rwlocks.find(uid);
            if (ok && it != emuenv.kernel.rwlocks.end() && it->second) {
                it->second->state = static_cast<RWLockState>(state);
                it->second->owners = std::move(owners);
            }
        }

        ok = ok && xr.get(count);
        for (uint32_t i = 0; ok && i < count; i++) {
            SceUID uid = 0;
            uint32_t used = 0;
            ok = xr.get(uid) && xr.get(used) && xr.need(used);
            if (!ok)
                break;
            const uint8_t *const bytes = xr.data + xr.pos;
            xr.pos += used;
            const auto it = emuenv.kernel.msgpipes.find(uid);
            if (it != emuenv.kernel.msgpipes.end() && it->second) {
                ByteRingBuffer &buffer = it->second->data_buffer;
                std::vector<uint8_t> discard(buffer.Used());
                buffer.Remove(discard.data(), discard.size());
                buffer.Insert(bytes, used);
            }
        }

        const uint64_t now = sync_timer_clock();
        ok = ok && xr.get(count);
        for (uint32_t i = 0; ok && i < count; i++) {
            SceUID uid = 0;
            uint8_t flags = 0;
            uint64_t interval = 0;
            int64_t next = 0, time = 0;
            ok = xr.get(uid) && xr.get(flags) && xr.get(interval) && xr.get(next) && xr.get(time);
            const auto it = emuenv.kernel.timers.find(uid);
            if (ok && it != emuenv.kernel.timers.end() && it->second) {
                Timer &timer = *it->second;
                timer.is_started = flags & 1;
                timer.is_repeat = flags & 2;
                timer.is_pulse = flags & 4;
                timer.event_set = flags & 8;
                timer.event_interval = interval;
                timer.next_event = next == std::numeric_limits<int64_t>::max() ? std::numeric_limits<uint64_t>::max() : now + next;
                timer.time = now + time;
            }
        }

        if (!ok)
            LOG_ERROR("Savestate: the SYN2 chunk is corrupt; rwlocks, message pipes and timers may not all be restored");
    }

    // --- restore the guest's file table ----------------------------------------------------
    // With the file table recorded, make every descriptor the guest holds refer to the file it
    // referred to when the state was taken, reopening it under that number if this session has
    // something else there or nothing at all. Without the table (older states), fall back to
    // seeking by number, which is only right when the two sessions opened files identically.
    uint32_t files_restored = 0, files_gone = 0, files_reopened = 0;
    if (ftbl_data) {
        Reader tr{ ftbl_data, ftbl_size, 0 };
        int32_t saved_next_fd = 0;
        uint32_t count = 0;
        if (!tr.get(saved_next_fd) || !tr.get(count))
            return Result::fail("corrupt file table chunk");
        for (uint32_t i = 0; i < count; i++) {
            SceUID fd = 0;
            int32_t flags = 0;
            int64_t at = 0;
            std::string path;
            if (!tr.get(fd) || !tr.get(flags) || !tr.get(at) || !tr.get_string(path))
                return Result::fail("corrupt file table chunk");

            auto it = emuenv.io.std_files.find(fd);
            if (it != emuenv.io.std_files.end() && !it->second.can_write_file() && path == it->second.get_vita_loc()) {
                if (it->second.seek(static_cast<SceOff>(at), SCE_SEEK_SET))
                    files_restored++;
                else
                    files_gone++;
                continue;
            }

            // The number means something else in this session, or nothing. A writable file
            // there is left alone: it is somebody's save data, and closing it is worse than one
            // stream reading from the wrong place.
            if (it != emuenv.io.std_files.end() && it->second.can_write_file()) {
                LOG_WARN("Savestate: descriptor {} is {} (writable) here but was {} in the state; leaving it",
                    fd, it->second.get_vita_loc(), path);
                files_gone++;
                continue;
            }
            const std::string was = (it != emuenv.io.std_files.end()) ? std::string(it->second.get_vita_loc()) : std::string("nothing");
            if (it != emuenv.io.std_files.end())
                emuenv.io.std_files.erase(it);

            const SceUID fresh = open_file(emuenv.io, path.c_str(), flags, emuenv.vita_fs_path, "savestate");
            if (fresh < 0) {
                LOG_WARN("Savestate: could not reopen {} for descriptor {}", path, fd);
                files_gone++;
                continue;
            }
            auto node = emuenv.io.std_files.extract(fresh);
            node.key() = fd;
            const auto placed = emuenv.io.std_files.insert(std::move(node));
            placed.position->second.seek(static_cast<SceOff>(at), SCE_SEEK_SET);
            LOG_INFO("Savestate: descriptor {} was {} here; reopened {} under it", fd, was, path);
            files_reopened++;
        }
        // Numbers the restored guest holds must never be handed out again.
        if (emuenv.io.next_fd < saved_next_fd)
            emuenv.io.next_fd = saved_next_fd;
    } else {
        Reader fr{ file_data, file_size, 0 };
        uint32_t count = 0;
        if (!fr.get(count))
            return Result::fail("corrupt file chunk");
        for (uint32_t i = 0; i < count; i++) {
            SceUID fd = 0;
            int64_t at = 0;
            if (!fr.get(fd) || !fr.get(at))
                return Result::fail("corrupt file chunk");
            const auto it = emuenv.io.std_files.find(fd);
            if (it == emuenv.io.std_files.end() || it->second.can_write_file()) {
                files_gone++;
                continue;
            }
            if (it->second.seek(static_cast<SceOff>(at), SCE_SEEK_SET))
                files_restored++;
            else
                files_gone++;
        }
    }

    // Restoring the counts above wrote numbers the waiting-thread queues know nothing about. A
    // waiter can now be satisfiable with nobody left to wake it, because semaphore_signal is the
    // only thing that ever wakes one -- which presents as the guest hanging with a correct-looking
    // frame on screen. Put the invariant back.
    reconcile_waiters_after_load(emuenv.kernel);

    // Host-side audio decoders hold a position inside the stream they were decoding. Guest memory
    // has just been rewound underneath them, so that position is now wrong and the next frame
    // unpacks nonsense -- observed as a storm of Atrac9 decode failures ending in a fault. Tell
    // them to resynchronise.
    ngs::on_savestate_loaded(emuenv.ngs, emuenv.mem);

    LOG_INFO("Savestate loaded: {} ({} regions, {}/{} threads, {} sync object(s), {} file position(s), {} reopened; {} sync and {} file gone)",
        path.string(), region_count, restored, thread_count, sync_restored, files_restored, files_reopened,
        sync_skipped, files_gone);
    return Result::ok();
}

} // namespace savestate
