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

#include <string>

struct EmuEnvState;

namespace savestate {

// What a state file currently covers. Guest RAM and CPU register state are captured; the
// kernel object graph, GPU state, open file handles and the audio graph are not. A state is
// therefore only meaningful to the same running process that produced it -- reloading one
// after restarting the emulator will not work, and the header records enough to refuse it.
inline constexpr uint32_t FORMAT_VERSION = 2;

struct Result {
    bool success = false;
    std::string reason;

    explicit operator bool() const {
        return success;
    }

    static Result ok() {
        return { true, {} };
    }
    static Result fail(std::string why) {
        return { false, std::move(why) };
    }
};

// <pref-path>/savestates/<TITLEID>/slot<N>.vst
fs::path slot_path(const EmuEnvState &emuenv, int slot);

// Both expect the guest to already be quiesced by the caller (see AppSessionPauseReason).
// Neither pauses or resumes on its own.
Result save(EmuEnvState &emuenv, const fs::path &path);
Result load(EmuEnvState &emuenv, const fs::path &path);

} // namespace savestate
