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

// A quicksave or a quickload the way a frontend runs one: catch a frame boundary, stop the guest and
// the renderer, save or load, and try again while the moment can't be taken. Shared by the Qt and
// the SDL frontends.

#include <savestate/savestate.h>

#include <app/session_controller.h>
#include <emuenv/state.h>
#include <gxm/functions.h>
#include <gxm/state.h>
#include <kernel/state.h>
#include <mem/state.h>
#include <renderer/functions.h>
#include <renderer/state.h>
#include <util/log.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <shared_mutex>
#include <thread>

namespace savestate {

namespace {

// Let the renderer finish what it already has before a save or load. With the guest stopped nothing
// new arrives. A batch left in the queue would run after a load against guest memory the load has
// rewritten -- measured as a crash in the texture upload path on Disgaea 4's second load. And the GPU
// has to finish too: the thread that waits for it writes notification values, sync timestamps and
// buffer syncs into guest memory as work completes. A write landing after the restore overwrote the
// restored guest's GPU notification counter, measured as Gravity Rush falling to 2 fps for good after
// a load. False if the GPU did not finish in time.
bool drain_renderer(EmuEnvState &emuenv) {
    renderer::State &renderer = *emuenv.renderer;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (renderer.command_buffer_queue.size() > 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // Without a GXM context nothing was submitted, and nothing would answer the wait
    if (gxm::capture_layout(emuenv.gxm, emuenv.mem).contexts.empty())
        return true;
    return renderer::wait_gpu_idle(renderer, 500);
}

// Runs action with the guest and the render thread stopped and guest memory to itself. A dedicated
// pause reason keeps this independent of the user's own pause, so taking a state while already
// paused does not resume the game underneath them.
Result run_quiesced(EmuEnvState &emuenv, app::AppSessionController &session, const std::function<Result()> &action) {
    const bool paused_before = session.is_paused();
    session.set_pause_reason(app::AppSessionPauseReason::Savestate, true);

    Result result = Result::fail("did not run");
    bool render_parked = false;
    std::unique_lock<std::shared_timed_mutex> guest_memory;
    if (const std::string blocker = quiesce(emuenv, 2000); !blocker.empty()) {
        result = Result::retry(blocker);
    } else if (!drain_renderer(emuenv)) {
        result = Result::retry("the GPU still has work in flight");
    } else if (!(render_parked = emuenv.renderer->park_render_thread(std::chrono::milliseconds(2000)))) {
        // The render thread is ours, not the guest's, and it reads guest memory continuously
        result = Result::fail("the render thread would not stop");
    } else if (const size_t batches = emuenv.renderer->command_buffer_queue.size(); batches > 0) {
        // Work built against guest memory as it is now
        result = Result::retry(fmt::format("the renderer still has {} batch(es) queued", batches));
    } else if (!(guest_memory = std::unique_lock<std::shared_timed_mutex>(emuenv.mem.savestate_lock, std::chrono::milliseconds(2000))).owns_lock()) {
        // An HLE call that reads guest memory while its thread reports waiting, like an audio output
        result = Result::retry("guest memory is still in use by an HLE call");
    } else {
        try {
            result = action();
        } catch (const std::exception &e) {
            result = Result::fail(e.what());
        }
    }
    // Get what just happened onto disk before the guest is let go. A load that goes wrong takes the
    // process with it a second or two later, and the async sink loses the last of the log.
    logging::flush();

    if (render_parked)
        emuenv.renderer->unpark_render_thread();
    guest_memory = {};
    release_threads(emuenv, paused_before);
    session.set_pause_reason(app::AppSessionPauseReason::Savestate, false);
    return result;
}

// Catches the moment the GXM display queue is empty, which is when no display callback is part-way
// through -- the queue thread pops an entry only after its callback has completed. A callback in
// flight has its progress in renderer-side timestamps while the SceGxmSyncObject those are checked
// against lives in guest memory, and a load rewinds one and not the other.
//
// It waits on the queue's own condition variable, notified on every pop, and holds the queue at the
// moment it is empty (Queue::wait_empty_and_hold), so the game finishes the scene it is drawing and
// stops at the queue's gate in sceGxmDisplayQueueAddEntry. The hold is released however the attempt
// ends: one left in place would block the guest there for good.
struct DisplayQueueHold {
    EmuEnvState *emuenv = nullptr;

    void release() {
        if (emuenv) {
            emuenv->gxm.display_queue.release_hold();
            emuenv = nullptr;
        }
    }

    ~DisplayQueueHold() {
        release();
    }
};

bool wait_for_display_queue_gap(EmuEnvState &emuenv, int timeout_ms, DisplayQueueHold &hold, const std::atomic<bool> &cancel) {
    GxmState &gxm = emuenv.gxm;
    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + std::chrono::milliseconds(timeout_ms);
    bool empty = false;
    while (!(empty = gxm.display_queue.wait_empty_and_hold(std::chrono::milliseconds(100)))
        && !cancel && std::chrono::steady_clock::now() < deadline) {
    }
    if (!empty)
        return false;
    hold.emuenv = &emuenv;
    // Pausing straight away, attempts a frame apart kept landing inside the scene: Disgaea 4 refused
    // 60 in a row, five times running
    const auto scene_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (gxm::scene_in_progress(gxm, emuenv.mem) && std::chrono::steady_clock::now() < scene_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started);
    LOG_INFO("Savestate: caught the display queue empty after {} ms and held it", waited.count());
    return true;
}

// Tries action at frame boundaries until it works, can't work, or 60 attempts went by. A refused
// attempt has changed nothing, and the moments a save or load refuses pass within a frame or two: a
// thread just past its call, a frame queued again between the gap and the pause.
Result attempt(EmuEnvState &emuenv, app::AppSessionController &session, const char *what, const std::atomic<bool> &cancel, int pause_ms, const std::function<Result()> &action) {
    constexpr int max_attempts = 60;
    Result last = Result::fail("did not run");
    for (int attempt = 1; attempt <= max_attempts; attempt++) {
        if (cancel)
            return Result::fail("cancelled");
        DisplayQueueHold hold;
        if (!wait_for_display_queue_gap(emuenv, 10000, hold, cancel)) {
            // A queue that never empties is a display callback that never finishes
            return cancel ? Result::fail("cancelled") : Result::retry("could not catch a frame boundary");
        }
        last = run_quiesced(emuenv, session, action);
        hold.release();
        if (last || !last.retryable)
            return last;
        LOG_INFO("Savestate: {} attempt {}/{} refused ({})", what, attempt, max_attempts, last.reason);
        // Let the guest move on before the next attempt lands in the same moment
        std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
    }
    LOG_ERROR("{} gave up after {} attempts: {}", what, max_attempts, last.reason);
    return last;
}

} // namespace

Result quicksave(EmuEnvState &emuenv, app::AppSessionController &session, int slot, const std::atomic<bool> &cancel) {
    const fs::path path = slot_path(emuenv, slot);
    return attempt(emuenv, session, "quicksave", cancel, 16, [&] { return save(emuenv, path); });
}

Result quickload(EmuEnvState &emuenv, app::AppSessionController &session, int slot, const std::atomic<bool> &cancel) {
    const fs::path path = slot_path(emuenv, slot);
    return attempt(emuenv, session, "quickload", cancel, 33, [&] { return load(emuenv, path); });
}

std::optional<std::thread> start_roundtrip_from_env(EmuEnvState &emuenv, app::AppSessionController &session, const std::atomic<bool> &cancel) {
    // ROUNDTRIP saves and loads back; SAVE only saves, and LOAD only loads, a state an earlier run
    // saved, which checks a load into a session that never had the state's threads and objects
    const char *after = nullptr;
    bool do_save = true, do_load = true;
    if ((after = std::getenv("VITA3K_SNAPSHOT_ROUNDTRIP"))) {
    } else if ((after = std::getenv("VITA3K_SNAPSHOT_SAVE"))) {
        do_load = false;
    } else if ((after = std::getenv("VITA3K_SNAPSHOT_LOAD"))) {
        do_save = false;
    } else {
        return std::nullopt;
    }
    // A list of times, seconds after now, like 60,90,120: one round trip at each
    std::vector<std::chrono::milliseconds> times;
    for (const char *at = after; *at;) {
        times.emplace_back(static_cast<int64_t>(std::atof(at) * 1000));
        const char *const comma = std::strchr(at, ',');
        if (!comma)
            break;
        at = comma + 1;
    }
    return std::thread([&emuenv, &session, &cancel, times, do_save, do_load] {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < times.size(); i++) {
            while (!cancel && std::chrono::steady_clock::now() - start < times[i])
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (cancel)
                return;
            LOG_INFO("Savestate: round trip {}/{} starts", i + 1, times.size());
            if (do_save) {
                if (const Result saved = quicksave(emuenv, session, 0, cancel); !saved) {
                    LOG_ERROR("Savestate: round trip {} could not save: {}", i + 1, saved.reason);
                    continue;
                }
            }
            if (do_load) {
                if (const Result loaded = quickload(emuenv, session, 0, cancel); !loaded) {
                    LOG_ERROR("Savestate: round trip {} could not load: {}", i + 1, loaded.reason);
                    continue;
                }
            }
            LOG_INFO("Savestate: round trip {} done", i + 1);
        }
    });
}

} // namespace savestate
