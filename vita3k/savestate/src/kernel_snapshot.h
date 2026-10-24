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

// The kernel's part of a savestate: its threads and its objects, as recipes. Each is written as
// what it was made from (the public creation fields) and what a game can see about it now (the
// Vita's own info structs), and read back by making it again through its constructor and driving it
// to that state with the same public operations a game uses. Nothing private is read or written,
// so a kernel change that keeps the Vita's behaviour leaves this alone.
//
// A thread blocked in a call is not resumed inside it. A load takes every thread of the session
// out of its call, and points each thread the state had blocked back at the svc of its import
// stub, so it makes the call again.

#pragma once

#include <cpu/common.h>
#include <kernel/types.h>
#include <mem/util.h>
#include <savestate/savestate.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

struct EmuEnvState;

namespace savestate::kernel {

// Reads a 32-bit word of guest memory, or nothing if the address is not mapped
using WordReader = std::function<std::optional<uint32_t>(Address)>;

// Why ctx can't be pointed back at the import call it is in, or empty if it can
std::string why_not_at_a_call(const WordReader &read, const CPUContext &ctx);

enum class Resume : uint8_t {
    // Dormant: nothing to resume
    dormant,
    // Stopped at its gate, in its own code or on its way back from a call: goes on from its registers
    registers,
    // Blocked in a call: makes the call again from its stub
    restart_call,
};

struct ThreadRecord {
    // Recipe
    SceUID id = 0;
    std::string name;
    Address entry_point = 0;
    uint32_t attr = 0;
    int32_t stack_size = 0;
    Address stack = 0;
    Address tls = 0;
    int32_t init_priority = 0;
    int32_t init_affinity_mask = 0;
    CPUContext init_cpu_ctx;
    // Observed
    Resume resume = Resume::dormant;
    uint32_t wait_type = 0;
    int32_t priority = 0;
    int32_t affinity_mask = 0;
    int32_t vm_suspends = 0;
    CPUContext ctx;
    uint32_t tpidrurw = 0;
};

// One kernel object: its class (UidClass), its UID and its record, whose layout the class decides
struct ObjectRecord {
    uint32_t uid_class = 0;
    SceUID uid = 0;
    std::vector<uint8_t> data;
};

struct Snapshot {
    std::vector<ThreadRecord> threads;
    std::vector<ObjectRecord> objects;
};

// Stops every guest thread, then waits until each one is dormant, at its gate, or blocked in a call.
// Empty once they are, or what wouldn't stop. Undone by release_threads, also when it fails.
std::string quiesce(EmuEnvState &emuenv, int timeout_ms);
void release_threads(EmuEnvState &emuenv);

// Records every thread and object of a quiesced guest, or refuses a moment no load could restart,
// retryably. After guest memory is copied: a message pipe's contents are read through guest memory.
Result capture(EmuEnvState &emuenv, Snapshot &snapshot);

void write(const Snapshot &snapshot, std::vector<uint8_t> &threads, std::vector<uint8_t> &objects);
bool read(const uint8_t *threads, size_t threads_size, const uint8_t *objects, size_t objects_size, Snapshot &snapshot);

// What a load does with the session's threads
struct LoadPlan {
    // In the state, and kept from the session as they are
    std::vector<SceUID> keep;
    // In the state, and made again: missing from the session, or different in it
    std::vector<SceUID> create;
    // Session threads that go: not in the state, or made again
    std::vector<SceUID> remove;
};

// Decides how to bring the session's threads to the state's, and checks that every thread the state
// had blocked can make its call again, against the state's memory. Touches nothing.
Result plan(EmuEnvState &emuenv, const Snapshot &snapshot, const WordReader &state_word, LoadPlan &plan);

// Before guest memory is restored: takes every session thread out of its call to its gate, ends the
// threads the plan removes, and deletes every kernel object.
void stop_session(EmuEnvState &emuenv, const LoadPlan &plan);

// After guest memory is restored: makes the threads and objects of the state, and gives every thread
// its registers. The threads stay held until the caller releases them.
void rebuild(EmuEnvState &emuenv, const Snapshot &snapshot, const LoadPlan &plan);

} // namespace savestate::kernel
