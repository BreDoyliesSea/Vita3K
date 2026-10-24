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

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <vector>

struct EmuEnvState;

namespace app {
class AppSessionController;
}

namespace savestate {

// The layout of a state file. 4 keeps the kernel's threads and objects as recipes (THR, OBJ) instead
// of copies of their fields.
inline constexpr uint32_t FORMAT_VERSION = 4;

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

// Stops every guest thread for a save or a load, and waits until each one is dormant, at its gate,
// or blocked in a call. Empty once they are, or what would not stop. Undone by release_threads(),
// which the caller calls however it went.
std::string quiesce(EmuEnvState &emuenv, int timeout_ms);
void release_threads(EmuEnvState &emuenv);

// Both expect quiesce() to have succeeded and MemState::savestate_lock to be held exclusively. The
// threads stay stopped until the caller releases them.
Result save(EmuEnvState &emuenv, const fs::path &path);
Result load(EmuEnvState &emuenv, const fs::path &path);

// The whole of a quicksave or a quickload, the way a frontend runs one: catches a frame boundary,
// stops the guest and the renderer, and tries again while the moment can't be taken, up to 60
// times. cancel stops it between attempts.
Result quicksave(EmuEnvState &emuenv, app::AppSessionController &session, int slot, const std::atomic<bool> &cancel);
Result quickload(EmuEnvState &emuenv, app::AppSessionController &session, int slot, const std::atomic<bool> &cancel);

// For tests: with VITA3K_SNAPSHOT_ROUNDTRIP=<seconds> set, a thread that quicksaves and then
// quickloads that state that long after now, so a run shows whether the guest goes on as if nothing
// had happened. VITA3K_SNAPSHOT_SAVE only saves, and VITA3K_SNAPSHOT_LOAD only loads the state an
// earlier run saved. Nothing without them. The caller sets cancel and joins the thread before the session ends.
std::optional<std::thread> start_roundtrip_from_env(EmuEnvState &emuenv, app::AppSessionController &session, const std::atomic<bool> &cancel);

} // namespace savestate
