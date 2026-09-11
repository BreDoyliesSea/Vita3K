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

#include <util/fs.h>

#include <cstdint>
#include <string>
#include <vector>

struct EmuEnvState;

namespace savestate {

// What a state file currently covers. Guest RAM and CPU register state are captured; the
// kernel object graph, GPU state, open file handles and the audio graph are not. A state is
// therefore only meaningful to the same running process that produced it -- reloading one
// after restarting the emulator will not work, and the header records enough to refuse it.
inline constexpr uint32_t FORMAT_VERSION = 3;

struct Result {
    bool success = false;
    std::string reason;
    // Whether trying again in a frame or two could succeed. The guest's thread topology drifts
    // and comes back, so a load refused for that is worth retrying; a state from another build or
    // another game never becomes loadable, and retrying one costs the guest a quiesce cycle per
    // attempt for nothing.
    bool retryable = false;

    explicit operator bool() const {
        return success;
    }

    static Result ok() {
        return { true, {}, false };
    }
    static Result fail(std::string why) {
        return { false, std::move(why), false };
    }
    static Result retry(std::string why) {
        return { false, std::move(why), true };
    }
};

// <pref-path>/savestates/<TITLEID>/slot<N>.vst
fs::path slot_path(const EmuEnvState &emuenv, int slot);

// One guest thread parked in a wait that only another thread can end -- the waits a load has to
// find again exactly as recorded. Waits that end on their own (a delay, an audio buffer) are left
// out, as the load leaves them out.
struct HeldWait {
    int32_t thread; // SceUID
    uint32_t nid;
    uint32_t args[3];

    bool operator==(const HeldWait &) const = default;
};
using WaitSignature = std::vector<HeldWait>; // sorted by thread

// Which threads are held in such a wait right now. Safe to call while the guest runs.
WaitSignature wait_signature(EmuEnvState &emuenv);

// Both expect the guest to already be quiesced by the caller (see AppSessionPauseReason) and
// MemState::savestate_lock to be held exclusively. Neither pauses or resumes on its own. load()
// relies on the lock beyond its own return: the threads it takes out of their calls wait on it
// before they continue, so they start when the caller lets go of it.
Result save(EmuEnvState &emuenv, const fs::path &path);
Result load(EmuEnvState &emuenv, const fs::path &path);

} // namespace savestate
