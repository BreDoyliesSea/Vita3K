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

// NIDs the restore path understands. A parked thread can only be carried across a load if its
// wait is one of these.
constexpr uint32_t NID_sceKernelWaitSema = 0x0C7B834BU;
constexpr uint32_t NID_sceKernelDelayThread = 0x4B675D05U;
constexpr uint32_t NID_sceAudioOutOutput = 0x02DB3F5FU;

// Waits that finish on their own in bounded real time whatever the guest does: a fixed delay, and
// an audio buffer the device drains. They need no matching across a load -- whichever one such a
// thread happens to be in, it will fall out of it shortly and rejoin the guest's own loop.
// A semaphore wait is different: it ends only when another thread signals, so a thread sitting in
// one is part of the state and has to still be sitting in the same one.
bool is_self_completing_wait(uint32_t nid) {
    return nid == NID_sceKernelDelayThread || nid == NID_sceAudioOutOutput;
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
        // memory or registers, so those threads cannot be restored by any snapshot of the
        // guest. Report how many there are: it decides whether restricting save points to
        // "nothing is blocked" is even achievable.
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
        LOG_WARN_IF(waiting > 0,
            "Savestate: {} thread(s) are blocked inside an HLE call; their wait state cannot be captured "
            "and they will resume as if the wait had returned",
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
            return Result::fail(fmt::format("a frame is still in flight ({} display queue entries)", pending));
    }

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
        default:
            LOG_WARN("Savestate: ignoring unknown chunk 0x{:08X}", tag);
            break;
        }
    }

    if (!mem_packed || !cpu_data || !wait_data || !sync_data || !file_data)
        return Result::fail("state is missing a required chunk");

    // --- gate: is the guest still parked the way the state expects? --------------------------
    // Run before anything is written. Restoring guest memory to a snapshot while the threads sit
    // in different waits than they did is what makes a load look successful and then fall over
    // half a minute later; refusing is better than that, and leaves the session untouched.
    struct ParkedWait {
        SceUID thread;
        uint32_t nid;
        uint32_t args[4];
    };
    std::vector<ParkedWait> parked;
    {
        Reader wr{ wait_data, wait_size, 0 };
        uint32_t count = 0;
        if (!wr.get(count))
            return Result::fail("corrupt wait chunk");
        parked.reserve(count);
        for (uint32_t i = 0; i < count; i++) {
            ParkedWait w{};
            if (!wr.get(w.thread) || !wr.get(w.nid))
                return Result::fail("corrupt wait chunk");
            for (uint32_t &arg : w.args) {
                if (!wr.get(arg))
                    return Result::fail("corrupt wait chunk");
            }
            parked.push_back(w);
        }
    }

    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);

        // Every semaphore wait recorded in the state must still be held by the same thread on the
        // same semaphore. needCount and the timeout pointer are compared too, cheaply, since they
        // are already to hand.
        for (const ParkedWait &w : parked) {
            if (is_self_completing_wait(w.nid))
                continue;
            if (w.nid != NID_sceKernelWaitSema) {
                const char *const name = w.nid ? import_name(w.nid) : nullptr;
                return Result::fail(fmt::format("thread {} was parked in {}, which this build cannot restore",
                    w.thread, name ? name : fmt::format("{}", log_hex(w.nid))));
            }
            const auto it = emuenv.kernel.threads.find(w.thread);
            if (it == emuenv.kernel.threads.end() || !it->second)
                return Result::fail(fmt::format("thread {} from the state no longer exists", w.thread));
            const auto &thread = it->second;
            const uint32_t now_nid = thread->current_import_nid.load(std::memory_order_relaxed);
            if (thread->status != ThreadStatus::wait
                || now_nid != w.nid
                || thread->current_import_args[0] != w.args[0]
                || thread->current_import_args[1] != w.args[1]
                || thread->current_import_args[2] != w.args[2]) {
                // Say which of the four things differs. "Has moved on" on its own does not
                // distinguish a thread that is off doing work from one that is parked in the
                // right call on the wrong object, and the two have different odds of coming back.
                const char *what = "is running";
                if (thread->status != ThreadStatus::wait)
                    what = "is not waiting";
                else if (now_nid != w.nid)
                    what = "is in a different call";
                else
                    what = "is waiting on a different object";
                return Result::retry(fmt::format(
                    "thread {} \"{}\" {} (state has it in semaphore {}, now {} on {})",
                    w.thread, thread->name, what, log_hex(w.args[0]),
                    now_nid ? import_name(now_nid) : "nothing", log_hex(thread->current_import_args[0])));
            }
        }

        // And the converse: a thread in a wait now that it was not in then is equally a mismatch,
        // because its host frame is one the snapshot knows nothing about.
        //
        // Matching on identity alone is not enough. A thread recorded in a self-completing wait --
        // say a 1 ms delay -- can be sitting in a semaphore wait by the time the load runs, and an
        // "is it in the list?" test passes that happily while the frame it is actually parked in
        // is one the state never saw. Require the recorded entry to describe the same wait.
        for (const auto &pair : emuenv.kernel.threads) {
            const auto &thread = pair.second;
            if (!thread || thread->status != ThreadStatus::wait)
                continue;
            const uint32_t nid = thread->current_import_nid.load(std::memory_order_relaxed);
            if (is_self_completing_wait(nid))
                continue;
            const auto recorded = std::find_if(parked.begin(), parked.end(),
                [&](const ParkedWait &w) { return w.thread == pair.first; });
            if (recorded == parked.end()) {
                return Result::retry(fmt::format("thread {} \"{}\" is waiting now but was not when the state was taken",
                    pair.first, thread->name));
            }
            if (recorded->nid != nid || recorded->args[0] != thread->current_import_args[0]
                || recorded->args[1] != thread->current_import_args[1]
                || recorded->args[2] != thread->current_import_args[2]) {
                const char *const now = import_name(nid);
                const char *const then = recorded->nid ? import_name(recorded->nid) : nullptr;
                return Result::retry(fmt::format("thread {} \"{}\" is in {} now but was in {} when the state was taken",
                    pair.first, thread->name, now ? now : "an unknown call", then ? then : "another call"));
            }
        }
    }

    std::vector<uint8_t> mem_raw;
    if (!inflate_to(mem_packed, mem_packed_size, mem_raw, mem_raw_size))
        return Result::fail("guest memory chunk is corrupt");

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
    std::vector<std::pair<Address, uint32_t>> host_owned_ranges;
    gxm::collect_host_owned_ranges(emuenv.gxm, emuenv.mem, host_owned_ranges);
    ngs::collect_host_owned_ranges(emuenv.ngs, emuenv.mem, host_owned_ranges);

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

    if (freed > 0 || reallocated > 0 || left_alone > 0)
        LOG_INFO("Savestate: address space reconciled ({} released, {} restored, {} left alone)",
            freed, reallocated, left_alone);
    if (unrecoverable > 0)
        LOG_ERROR("Savestate: {} region(s) could not be re-allocated; the guest address space does not match the state", unrecoverable);

    if (skipped > 0)
        LOG_WARN("Savestate: {} region(s) are still not mapped and were skipped", skipped);

    if (host_owned_ranges_skipped > 0)
        LOG_INFO("Savestate: stepped over {} host-owned range(s) inside restored regions", host_owned_ranges_skipped);

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
    Reader cr{ cpu_data, cpu_size, 0 };
    uint32_t thread_count = 0;
    if (!cr.get(thread_count))
        return Result::fail("corrupt cpu chunk");

    uint32_t restored = 0;
    uint32_t missing = 0;
    {
        const std::lock_guard<std::mutex> lock(emuenv.kernel.mutex);
        for (uint32_t i = 0; i < thread_count; i++) {
            SceUID id = 0;
            CPUContext ctx;
            if (!cr.get(id) || !cr.get(ctx))
                return Result::fail("corrupt cpu chunk");

            const auto it = emuenv.kernel.threads.find(id);
            if (it == emuenv.kernel.threads.end() || !it->second || !it->second->cpu) {
                missing++;
                continue;
            }
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

    if (missing > 0)
        LOG_WARN("Savestate: {} thread(s) in the state no longer exist; their register state was dropped", missing);

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

    // --- restore read-only file positions ---------------------------------------------------
    // A descriptor that has since been closed, or has become writable, is skipped rather than
    // forced.
    uint32_t files_restored = 0, files_gone = 0;
    {
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

    LOG_INFO("Savestate loaded: {} ({} regions, {}/{} threads, {} sync object(s), {} file position(s); {} sync and {} file gone)",
        path.string(), region_count, restored, thread_count, sync_restored, files_restored,
        sync_skipped, files_gone);
    return Result::ok();
}

} // namespace savestate
