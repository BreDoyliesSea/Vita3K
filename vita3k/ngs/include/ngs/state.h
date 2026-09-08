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

// The guest addresses of every NGS mempool: the System and Rack memspaces.
//
// These are guest allocations, but the objects placement-new'd into them are *host* objects --
// System, Rack and Voice carry std::vector buffers, a std::unique_ptr<std::mutex> and vtable
// pointers, all of which are host addresses. A savestate captures those bytes like any other
// guest memory, and restoring them into a different process installs pointers belonging to the
// process that took the state. Walking them then faults immediately; measured as a read
// violation in on_savestate_loaded on the first `for (Rack *rack : system->racks)`.
//
// Within one session the addresses still happen to be valid, which is why an in-session load
// survives this and a load taken before a restart does not. Callers use this to leave these
// regions alone.
void collect_host_owned_memspaces(State &ngs, const MemState &mem, std::vector<Address> &out);
} // namespace ngs
