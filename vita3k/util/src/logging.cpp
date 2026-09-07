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

#include <util/log.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <DbgHelp.h>
#endif

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dup_filter_sink.h>
#include <spdlog/sinks/msvc_sink.h>
#ifdef __ANDROID__
#include <spdlog/sinks/android_sink.h>
#else
#include <spdlog/sinks/stdout_color_sinks.h>
#endif

#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace logging {

static const fs::path &LOG_FILE_NAME = "vita3k.log";
static const char *LOG_PATTERN = "%^[%H:%M:%S.%e] |%L| [%!]: %v%$";
static constexpr size_t ASYNC_LOG_QUEUE_SIZE = 65536;
static std::vector<spdlog::sink_ptr> sinks;
static std::once_flag s_async_logging_once;

static std::function<void(std::string, int)> s_log_callback;
static std::mutex s_log_callback_mutex;

static void register_log_exception_handler();
static void rebuild_default_logger();

static void flush() {
    spdlog::details::registry::instance().flush_all();
}

template <typename Mutex>
class callback_sink final : public spdlog::sinks::base_sink<Mutex> {
protected:
    void sink_it_(const spdlog::details::log_msg &msg) override {
        std::function<void(std::string, int)> callback;
        {
            const std::lock_guard<std::mutex> lock(s_log_callback_mutex);
            callback = s_log_callback;
        }

        if (!callback)
            return;

        spdlog::memory_buf_t formatted;
        spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
        callback(fmt::to_string(formatted), static_cast<int>(msg.level));
    }

    void flush_() override {};
};

using callback_sink_mt = callback_sink<std::mutex>;

void set_log_callback(std::function<void(std::string, int)> cb) {
    const std::lock_guard<std::mutex> lock(s_log_callback_mutex);
    s_log_callback = std::move(cb);
}

ExitCode init(const Root &root_paths, bool use_stdout) {
    sinks.clear();
    if (use_stdout)
#ifdef __ANDROID__
        sinks.push_back(std::make_shared<spdlog::sinks::android_sink_mt>("Vita3K"));
#else
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
#endif

#ifndef __ANDROID__
    sinks.push_back(std::make_shared<callback_sink_mt>());
#endif

    if (add_sink(root_paths.get_log_path() / LOG_FILE_NAME) != Success)
        return InitConfigFailed;

    spdlog::set_error_handler([](const std::string &msg) {
        std::cerr << "spdlog error: " << msg << std::endl;
        assert(0);
    });

#ifdef _WIN32
    // set console codepage to UTF-8
    SetConsoleOutputCP(65001);
    SetConsoleTitle("Vita3K PSVita Emulator");
#endif

#ifdef __ANDROID__
    // needed, otherwise the log file contains nothing
    spdlog::flush_on(spdlog::level::trace);
#endif

    register_log_exception_handler();

    static std::terminate_handler old_terminate = nullptr;
    old_terminate = std::set_terminate([]() {
        try {
            auto eptr = std::current_exception();
            if (eptr) {
                std::rethrow_exception(eptr);
            } else {
                LOG_CRITICAL("Unhandled 'std::terminate()' call");
            }
        } catch (const std::exception &e) {
            LOG_CRITICAL("Unhandled C++ exception. {}", e.what());
        } catch (...) {
            LOG_CRITICAL("Unhandled C++ exception. UNKNOWN");
        }
        flush();
        if (old_terminate)
            old_terminate();
    });
    return Success;
}

void set_level(spdlog::level::level_enum log_level) {
    spdlog::set_level(log_level);
}

ExitCode add_sink(const fs::path &log_path) {
    try {
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.generic_path().native(), true));
    } catch (const spdlog::spdlog_ex &ex) {
        std::cerr << "File log initialization failed: " << ex.what() << std::endl;
        return InitConfigFailed;
    }

#ifdef _MSC_VER
    sinks.push_back(std::make_shared<spdlog::sinks::msvc_sink_mt>());
#endif

    rebuild_default_logger();
    return Success;
}

void rebuild_default_logger() {
    std::call_once(s_async_logging_once, []() {
        spdlog::init_thread_pool(ASYNC_LOG_QUEUE_SIZE, 1);
    });

    auto duplicate_filter = std::make_shared<spdlog::sinks::dup_filter_sink_mt>(std::chrono::seconds(2));
    for (const auto &sink : sinks)
        duplicate_filter->add_sink(sink);

    auto logger = std::make_shared<spdlog::async_logger>(
        "vita3k logger",
        duplicate_filter,
        spdlog::thread_pool(),
        spdlog::async_overflow_policy::overrun_oldest);
    spdlog::set_default_logger(std::move(logger));
    spdlog::set_pattern(LOG_PATTERN);
}

// log exceptions and flush log file on exceptions
#ifdef _WIN32
// Base and extent of our own executable image, resolved once at startup by walking the PE
// headers. Doing it here rather than in the handler keeps the handler free of loader calls, which
// are not safe to make while a fault is being dispatched.
static uintptr_t s_exe_base = 0;
static size_t s_exe_size = 0;

// Symbol lookup for crash addresses. Initialised once at startup, because SymInitialize walks the
// module list and must not be called while a fault is being dispatched. SymFromAddr itself is not
// documented as safe inside a handler either, but by the time this runs the process is already
// going down and a function name is worth more than the small risk of not getting one.
static bool s_symbols_ready = false;

static void init_symbols() {
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    s_symbols_ready = SymInitialize(GetCurrentProcess(), nullptr, TRUE) != FALSE;
}

static std::string symbol_for(uintptr_t address) {
    if (!s_symbols_ready)
        return {};
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME] = {};
    auto *symbol = reinterpret_cast<SYMBOL_INFO *>(buffer);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    DWORD64 displacement = 0;
    if (!SymFromAddr(GetCurrentProcess(), address, &displacement, symbol))
        return {};
    return fmt::format(" [{}+{}]", symbol->Name, displacement);
}

static void resolve_own_image() {
    const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr));
    if (!base)
        return;
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return;
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE)
        return;
    s_exe_base = base;
    s_exe_size = nt->OptionalHeader.SizeOfImage;
}

// Where the faulting instruction was, not just what it touched. An address inside our own image
// means host code followed a bad pointer; an address outside every image is almost always
// dynarmic's generated code, which means the guest is executing something it should not be. The
// two have completely different causes and the fault address alone does not distinguish them.
static void log_fault_origin(PEXCEPTION_POINTERS pExp) {
    const auto pc = reinterpret_cast<uintptr_t>(pExp->ExceptionRecord->ExceptionAddress);
    const std::string symbol = symbol_for(pc);
    if (s_exe_base && pc >= s_exe_base && pc < s_exe_base + s_exe_size) {
        LOG_CRITICAL("  faulting instruction at Vita3K.exe+{}{} (thread {})",
            log_hex(static_cast<uint64_t>(pc - s_exe_base)), symbol, GetCurrentThreadId());
    } else {
        LOG_CRITICAL("  faulting instruction at {}{}, outside our image - JIT-generated or another "
                     "module (thread {})",
            log_hex(static_cast<uint64_t>(pc)), symbol, GetCurrentThreadId());
    }
}

static LONG WINAPI exception_handler(PEXCEPTION_POINTERS pExp) noexcept {
    const unsigned ec = pExp->ExceptionRecord->ExceptionCode;
    switch (ec) {
    case EXCEPTION_ACCESS_VIOLATION:
        LOG_CRITICAL("Exception EXCEPTION_ACCESS_VIOLATION ({}). ", log_hex(ec));
        log_fault_origin(pExp);
        switch (pExp->ExceptionRecord->ExceptionInformation[0]) {
        case 0:
            LOG_CRITICAL("Read violation at address {}.", log_hex(pExp->ExceptionRecord->ExceptionInformation[1]));
            break;
        case 1:
            LOG_CRITICAL("Write violation at address {}.", log_hex(pExp->ExceptionRecord->ExceptionInformation[1]));
            break;
        case 8:
            LOG_CRITICAL("DEP violation at address {}.", log_hex(pExp->ExceptionRecord->ExceptionInformation[1]));
            break;
        default:
            break;
        }
        break;
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        LOG_CRITICAL("Exception EXCEPTION_ARRAY_BOUNDS_EXCEEDED ({}). ", log_hex(ec));
        break;
    case EXCEPTION_DATATYPE_MISALIGNMENT:
        LOG_CRITICAL("Exception EXCEPTION_DATATYPE_MISALIGNMENT ({}). ", log_hex(ec));
        break;
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        LOG_CRITICAL("Exception EXCEPTION_FLT_DIVIDE_BY_ZERO ({}). ", log_hex(ec));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        LOG_CRITICAL("Exception EXCEPTION_ILLEGAL_INSTRUCTION ({}). ", log_hex(ec));
        break;
    case EXCEPTION_IN_PAGE_ERROR:
        LOG_CRITICAL("Exception EXCEPTION_IN_PAGE_ERROR ({}). ", log_hex(ec));
        break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        LOG_CRITICAL("Exception EXCEPTION_INT_DIVIDE_BY_ZERO ({}). ", log_hex(ec));
        break;
    case EXCEPTION_PRIV_INSTRUCTION:
        LOG_CRITICAL("Exception EXCEPTION_PRIV_INSTRUCTION ({}). ", log_hex(ec));
        break;
    case EXCEPTION_STACK_OVERFLOW:
        LOG_CRITICAL("Exception EXCEPTION_STACK_OVERFLOW ({}). ", log_hex(ec));
        break;
    default:
        return EXCEPTION_CONTINUE_SEARCH;
    }
    flush();
    return EXCEPTION_CONTINUE_SEARCH;
}

void register_log_exception_handler() {
    resolve_own_image();
    init_symbols();
    if (!AddVectoredExceptionHandler(0, exception_handler)) {
        LOG_CRITICAL("Failed to register an exception handler");
    }
}

#else
void register_log_exception_handler() {}
#endif
} // namespace logging
