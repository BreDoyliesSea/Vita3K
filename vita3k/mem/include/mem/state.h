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

#include <mem/allocator.h>
#include <mem/functions.h>
#include <mem/util.h>

#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <set>

struct AllocMemPage {
    uint32_t allocated : 4;
    uint32_t size : 28;
};

static_assert(sizeof(AllocMemPage) == 4);

typedef uint8_t *PagePtr;
typedef std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>> Memory;
typedef std::unique_ptr<AllocMemPage[]> AllocPageTable;
typedef std::unique_ptr<PagePtr[]> PageTable;
typedef std::map<int, std::string> PageNameMap;

struct ProtectBlockInfo {
    uint32_t size = 0;
    ProtectCallback callback;
};

struct ProtectSegmentInfo {
    std::multimap<Address, ProtectBlockInfo> blocks;
    uint32_t size = 0;
    MemPerm perm = MemPerm::None;

    explicit ProtectSegmentInfo() = default;
    explicit ProtectSegmentInfo(uint32_t size, MemPerm perm)
        : size(size)
        , perm(perm) {
    }
};

typedef std::map<Address, ProtectSegmentInfo, std::greater<>> ProtectSegmentTrees;

struct MemExternalMapping {
    Address address;
    uint32_t size;
};

struct MemState {
    std::mutex generation_mutex;
    std::mutex protect_mutex;

    // --- savestate barrier -------------------------------------------------------------------
    // Held shared by anything that reads or writes guest memory outside the guest's own
    // execution, and exclusively by a savestate while it rewrites that memory.
    //
    // Pausing the guest is not enough on its own. pause_threads() suspends the threads that are
    // *running*; a thread parked in an HLE call is recorded as waiting and left alone -- and some
    // HLE calls do their real work while nominally waiting. sceAudioOutOutput is the clearest
    // case: it sets the thread's status to 'wait', then hands the guest's buffer straight to the
    // audio backend, which memcpy's out of it. A load rewriting guest memory during that window
    // faulted in memmove on the game's audio thread, repeatably.
    std::shared_timed_mutex savestate_lock;

    uint32_t host_page_size = 0;
    Memory memory;
    AllocPageTable alloc_table;
    BitmapAllocator allocator;
    ProtectSegmentTrees protect_tree;
    // Host pages currently carrying a protection set by protect_inner(). Lets the fault handler
    // tell "this page is protected and I must act" from "another thread already handled this and
    // unprotected it", which is otherwise indistinguishable once the segment has been erased
    // from protect_tree.
    //
    // Has its own mutex rather than reusing protect_mutex: protect_inner/unprotect_inner are
    // reached both from paths that already hold protect_mutex (add_protect,
    // handle_access_violation) and from paths that do not (kubridge). It is always the innermost
    // lock, so there is no ordering hazard.
    std::mutex protected_pages_mutex;
    std::set<Address> protected_pages;

    PageNameMap page_name_map;

    bool use_page_table = false;
    PageTable page_table;
    std::map<uint64_t, MemExternalMapping, std::greater<>> external_mapping;
};
