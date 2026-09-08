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
} // namespace ngs
