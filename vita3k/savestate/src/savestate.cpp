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
#include <io/state.h>
#include <kernel/state.h>
#include <kernel/thread/thread_state.h>
#include <mem/functions.h>
#include <mem/state.h>
#include <nids/functions.h>
#include <util/log.h>

#include <miniz.h>

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

// A state records the emulator build that produced it. Guest memory is full of pointers into
// host-side structures whose layout this binary fixes, so a state from a different build is
// not merely stale, it is actively dangerous. Refuse it rather than crash confusingly later.
std::string build_identity() {
    return std::string(app_version) + "-" + std::to_string(app_number);
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
        default:
            LOG_WARN("Savestate: ignoring unknown chunk 0x{:08X}", tag);
            break;
        }
    }

    if (!mem_packed || !cpu_data)
        return Result::fail("state is missing a required chunk");

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

    std::map<Address, uint32_t> live;
    {
        const std::lock_guard<std::mutex> alloc_lock(emuenv.mem.generation_mutex);
        for_each_allocation(emuenv.mem, [&](Address addr, uint32_t size) {
            live.emplace(addr, size);
        });
    }

    std::map<Address, uint32_t> wanted;
    for (const auto &region : saved)
        wanted.emplace(region.addr, region.size);

    // free() and try_alloc_at() take the allocator lock themselves, so these run unlocked.
    uint32_t freed = 0;
    for (const auto &entry : live) {
        const auto it = wanted.find(entry.first);
        if (it == wanted.end() || it->second != entry.second) {
            free(emuenv.mem, entry.first);
            freed++;
        }
    }

    uint32_t reallocated = 0;
    uint32_t unrecoverable = 0;
    for (const auto &entry : wanted) {
        const auto it = live.find(entry.first);
        if (it != live.end() && it->second == entry.second)
            continue;
        if (try_alloc_at(emuenv.mem, entry.first, entry.second, "savestate") == 0) {
            unrecoverable++;
            continue;
        }
        reallocated++;
    }

    if (freed > 0 || reallocated > 0)
        LOG_INFO("Savestate: address space reconciled ({} freed, {} restored)", freed, reallocated);
    if (unrecoverable > 0)
        LOG_ERROR("Savestate: {} region(s) could not be re-allocated; the guest address space does not match the state", unrecoverable);

    // --- restore memory ---------------------------------------------------------------------
    uint32_t skipped = 0;
    {
        const std::lock_guard<std::mutex> alloc_lock(emuenv.mem.generation_mutex);
        for (const auto &region : saved) {
            if (!Ptr<uint8_t>(region.addr).valid(emuenv.mem)) {
                skipped++;
                continue;
            }
            std::memcpy(emuenv.mem.memory.get() + region.addr, mem_raw.data() + region.offset, region.size);
        }
    }
    if (skipped > 0)
        LOG_WARN("Savestate: {} region(s) are still not mapped and were skipped", skipped);

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

    LOG_INFO("Savestate loaded: {} ({} regions, {}/{} threads restored)",
        path.string(), region_count, restored, thread_count);
    return Result::ok();
}

} // namespace savestate
