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

#include <mem/ptr.h>

#include <utility>
#include <vector>
#include <cstdint>
#include <string>

struct MemState;

namespace ngs {
struct VoiceDefinition;
struct System;

struct State {
    Ptr<VoiceDefinition> definitions;
    std::vector<System *> systems;
};

bool init(State &ngs, MemState &mem);
void deinit(State &ngs, MemState &mem);

// Tell every live voice module that guest memory has been rewound under it, so host-side decoder
// positions can resynchronise. See ModuleRuntimeState::on_savestate_loaded.
void on_savestate_loaded(State &ngs, const MemState &mem);

// Guest byte ranges that hold host objects: the System, its Racks, and their Voices.
//
// These are constructed into guest memory but they are *host* objects -- std::vector buffers, a
// std::unique_ptr<std::mutex>, vtable pointers, all host addresses. A savestate captures those
// bytes like any other guest memory, and restoring a state taken by a different process installs
// pointers belonging to the process that took it. Walking them then faults immediately; measured
// as a read violation on the first `for (Rack *rack : system->racks)`.
//
// Only the objects, not the pool around them: the guest's own parameter buffers share that pool
// and must be rewound with the rest of guest memory. Collect before anything is overwritten.
// See gxm::collect_host_owned_ranges for the same problem in GXM.
void collect_host_owned_ranges(State &ngs, const MemState &mem,
    std::vector<std::pair<Address, uint32_t>> &out);

// --- moving NGS objects to where a savestate had them ------------------------------------------
// See the definitions in ngs.cpp for why this exists and what it relies on.

// Where each system, rack and voice was, in ngs.systems / racks / voices order. An address of 0
// is an empty slot.
struct SavedLayout {
    struct RackEntry {
        Address addr = 0;
        std::vector<Address> voices;
    };
    struct SystemEntry {
        Address addr = 0;
        std::vector<RackEntry> racks;
    };
    Address definitions = 0;
    std::vector<SystemEntry> systems;
};

SavedLayout capture_layout(State &ngs, const MemState &mem);

// Where each voice is in what it plays. The Voice objects are kept across a load (see
// collect_host_owned_ranges), and so was their playback: the restored guest then got the
// end-of-data callbacks of sounds it had not started yet, and Borderlands 2's audio thread read
// through the null it had for them. A load puts this back on the voices it kept instead.
struct SavedVoice {
    struct Module {
        std::vector<uint8_t> guest_state;
        int32_t loop_count = 0;
    };
    Address addr = 0;
    uint32_t state = 0;
    bool is_pending = false;
    bool is_paused = false;
    bool is_keyed_off = false;
    uint32_t frame_count = 0;
    std::vector<Module> modules;
    // The patches it feeds, per output port. The Patch objects are guest memory and come back with
    // the state; the voice's list of them did not, and mixing then followed this session's list
    // into whatever the restored memory held there.
    std::vector<std::vector<Address>> patches;
};
std::vector<SavedVoice> capture_voices(State &ngs, const MemState &mem);
// The number of voices put back. A voice the state does not describe, or describes with
// different modules, is left as it is.
uint32_t restore_voices(State &ngs, const MemState &mem, const std::vector<SavedVoice> &saved);

// Empty if the live objects can be moved onto `saved`, otherwise why not. Touches nothing.
std::string check_relocatable(State &ngs, const MemState &mem, const SavedLayout &saved);

// True while any scheduler is locked or mid-update; a load should come back a moment later.
bool scheduler_busy(State &ngs);

struct RelocationStash {
    struct Block {
        Address from;
        Address to;
        std::vector<uint8_t> bytes;
    };
    struct Move {
        Address from;
        uint32_t size;
        Address to;
    };
    std::vector<Block> blocks; // the objects themselves, carried as bytes
    std::vector<Move> moves; // the pools they sit in
};

// Copy the live objects aside. Call before guest memory is overwritten or reallocated.
RelocationStash stash_for_relocation(State &ngs, const MemState &mem, const SavedLayout &saved);

// After guest memory has been restored: put the stashed objects at their saved addresses and
// rewire every pointer between them. Returns how many pools actually moved.
uint32_t relocate_after_restore(State &ngs, MemState &mem, const RelocationStash &stash);
} // namespace ngs
