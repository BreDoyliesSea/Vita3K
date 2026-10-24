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

#ifndef queue_h
#define queue_h

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

template <typename T>
class Queue {
public:
    // default value: unlimited
    unsigned int maxPendingCount_ = -1;

    std::unique_ptr<T> top(const int ms = 0) {
        T item{ T() };
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            if (ms == 0) {
                while (!aborted && queue_.empty()) {
                    condempty_.wait(mlock);
                }
            } else {
                if (queue_.empty()) {
                    condempty_.wait_for(mlock, std::chrono::microseconds(ms));
                }
            }
            if (aborted || queue_.empty()) {
                return {};
            }

            item = queue_.front();
        }
        return std::make_unique<T>(item);
    }

    std::unique_ptr<T> pop(const int ms = 0) {
        T item{ T() };
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            if (ms == 0) {
                while (!aborted && queue_.empty()) {
                    condempty_.wait(mlock);
                }
            } else {
                if (queue_.empty()) {
                    condempty_.wait_for(mlock, std::chrono::microseconds(ms));
                }
            }
            if (aborted || queue_.empty()) {
                return {};
            }

            item = queue_.front();
            queue_.pop();
        }
        cond_.notify_all();
        return std::make_unique<T>(item);
    }

    void push(const T &item) {
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            while (!aborted && queue_.size() == maxPendingCount_) {
                cond_.wait(mlock);
            }
            if (aborted) {
                return;
            }
            queue_.push(item);
        }
        condempty_.notify_one();
    }

    size_t size() {
        return queue_.size();
    }

    void wake() {
        condempty_.notify_all();
    }

    void abort() {
        aborted = true;
        // A hold must not outlive the queue it holds: shutting down while one is in place would
        // leave a guest thread waiting in push() for a release that is no longer coming.
        {
            std::lock_guard<std::mutex> mlock(mutex_);
            held_ = false;
        }
        condempty_.notify_all();
        cond_.notify_all();
    }

    bool is_aborted() const {
        return aborted.load(std::memory_order_relaxed);
    }

    void reset() {
        std::queue<T> empty;
        std::swap(queue_, empty);
        aborted = false;
        held_ = false;
    }

    void wait_empty() {
        std::unique_lock<std::mutex> mlock(mutex_);
        cond_.wait(mlock, [&]() { return aborted || queue_.empty(); });
    }

    // Bounded form, for callers that must not block forever if the queue never drains.
    // cond_ is notified on every pop, so this wakes inside the window the queue is empty rather
    // than having to poll for it -- that window can be well under a millisecond.
    bool wait_empty_for(const std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> mlock(mutex_);
        return cond_.wait_for(mlock, timeout, [&]() { return aborted || queue_.empty(); });
    }

    // The same, but the queue stays empty until release_hold(): a producer waits at its gate
    // (wait_while_held, at the top of sceGxmDisplayQueueAddEntry); push() itself does not look at
    // the hold, so a producer already past the gate still queues, and the caller has to check the
    // queue again once the guest is stopped. A savestate needs the empty moment to last long
    // enough to stop the guest, and merely observing it is not enough to get it. pop() notifies cond_, which wakes
    // both this wait and the guest thread blocked in push() waiting for the slot that just freed,
    // so whoever takes the mutex first decides; when a game keeps the queue full - every game
    // measured here runs it at depth 1 - the pusher usually wins and the next entry is in before
    // the guest can be stopped. Measured on Gravity Rush: 289 of 289 quickload attempts found one
    // entry again 24-38 ms after catching the queue empty, and 3 of 10 loads gave up after 60
    // attempts. Holding it closes that window instead of racing it. Taking the empty reading and
    // the hold under one lock is the point of this being one call.
    bool wait_empty_and_hold(const std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> mlock(mutex_);
        if (!cond_.wait_for(mlock, timeout, [&]() { return aborted || queue_.empty(); }))
            return false;
        held_ = true;
        return true;
    }

    void release_hold() {
        {
            std::lock_guard<std::mutex> mlock(mutex_);
            held_ = false;
        }
        cond_.notify_all();
    }

    bool is_held() {
        std::lock_guard<std::mutex> mlock(mutex_);
        return held_;
    }

    // Where a producer waits for room, instead of inside push(). push() blocks on a full queue too,
    // but by then the caller's own call has done its work, and a savestate load can only put a
    // thread back at the *start* of the call it is parked in. Waiting here - before that work -
    // keeps the producer restartable. It is also what makes a hold effective: a thread already
    // blocked inside push() is past every gate, and is released by the same pop() that hands the
    // savestate its gap. This is ordinary frame pacing, so it does not touch the thread's status.
    void wait_for_space(const std::function<bool()> &abandoned) {
        std::unique_lock<std::mutex> mlock(mutex_);
        cond_.wait(mlock, [&]() {
            return aborted || queue_.size() < maxPendingCount_ || (abandoned && abandoned());
        });
    }

    // Waiting out a hold is not ordinary pacing: the hold is kept across the guest being stopped,
    // so a thread waiting here has to report that it is waiting or the quiesce waits for a thread
    // that cannot move. Kept separate from wait_for_space precisely so that reporting happens only
    // at the few moments a savestate asks for, not on every frame.
    void wait_while_held(const std::function<bool()> &abandoned) {
        std::unique_lock<std::mutex> mlock(mutex_);
        cond_.wait(mlock, [&]() { return aborted || !held_ || (abandoned && abandoned()); });
    }

    // For a waiter's WaitRelease: ends wait_while_held without releasing the hold itself.
    void wake_holders() {
        cond_.notify_all();
    }

    // wait_empty(), but a savestate load can end it. Used by a double-buffered producer, which
    // waits for the frame it has just queued to be taken.
    void wait_empty_or_abandoned(const std::function<bool()> &abandoned) {
        std::unique_lock<std::mutex> mlock(mutex_);
        cond_.wait(mlock, [&]() { return aborted || queue_.empty() || (abandoned && abandoned()); });
    }

    Queue() = default;
    Queue(const Queue &) = delete; // disable copying
    Queue &operator=(const Queue &) = delete; // disable assignment

    std::mutex &get_mutex() {
        return mutex_;
    }

private:
    bool held_ = false;
    std::condition_variable cond_;
    std::condition_variable condempty_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::atomic<bool> aborted{ false };
};

#endif /* queue_h */
