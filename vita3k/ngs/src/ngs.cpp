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

#include <cpu/functions.h>
#include <kernel/state.h>

#include <ngs/modules/atrac9.h>
#include <ngs/state.h>
#include <ngs/system.h>
#include <util/lock_and_find.h>

#include <util/vector_utils.h>
#include <util/log.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <queue>
#include <string>
#include <type_traits>

namespace ngs {
Rack::Rack(System *mama, const Ptr<void> memspace, const uint32_t memspace_size)
    : MempoolObject(memspace, memspace_size)
    , system(mama) {}

System::System(const Ptr<void> memspace, const uint32_t memspace_size)
    : MempoolObject(memspace, memspace_size)
    , max_voices(0)
    , granularity(0)
    , sample_rate(0) {}

bool Patch::is_active() const {
    return output_sub_index != -1;
}

void VoiceInputManager::init(const uint32_t granularity, const uint16_t total_input) {
    inputs.resize(total_input);

    for (auto &input : inputs) {
        // FLTP and maximum channel count
        input.resize(granularity * 8);
    }

    reset_inputs();
}

void VoiceInputManager::reset_inputs() {
    for (auto &input : inputs) {
        std::fill(input.begin(), input.end(), 0);
    }
}

VoiceInputManager::PCMInput *VoiceInputManager::get_input_buffer_queue(const int32_t index) {
    if (index >= inputs.size()) {
        return nullptr;
    }

    return &inputs[index];
}

int32_t VoiceInputManager::receive(const MemState &mem, ngs::Patch *patch, const VoiceProduct &product) {
    Voice *source = patch->source.get(mem);
    Voice *dest = patch->dest.get(mem);
    if (!source || !dest) {
        return -1;
    }

    PCMInput *input = get_input_buffer_queue(patch->dest_index);

    if (!input) {
        return -1;
    }

    float *dest_buffer = reinterpret_cast<float *>(input->data());
    const float *data_to_mix_in = reinterpret_cast<const float *>(product.data);

    float volume_matrix[2][2];
    memcpy(volume_matrix, patch->volume_matrix, sizeof(volume_matrix));

    // we always use stereo internally, so make sure not to add too many channels
    if (source->rack->channels_per_voice == 1) {
        volume_matrix[1][0] = 0.0f;
        volume_matrix[1][1] = 0.0f;
    }

    if (dest->rack->channels_per_voice == 1) {
        volume_matrix[0][1] = 0.0f;
        volume_matrix[1][1] = 0.0f;
    }

    // Try mixing, also with the use of this volume matrix
    // Dest is our voice to receive this data.
    for (int32_t k = 0; k < dest->rack->system->granularity; k++) {
        dest_buffer[k * 2] = std::clamp(dest_buffer[k * 2] + data_to_mix_in[k * 2] * volume_matrix[0][0]
                + data_to_mix_in[k * 2 + 1] * volume_matrix[1][0],
            -1.0f, 1.0f);
        dest_buffer[k * 2 + 1] = std::clamp(dest_buffer[k * 2 + 1] + data_to_mix_in[k * 2] * volume_matrix[0][1] + data_to_mix_in[k * 2 + 1] * volume_matrix[1][1], -1.0f, 1.0f);
    }

    return 0;
}

ModuleData::ModuleData()
    : callback(0)
    , user_data(0)
    , is_bypassed(false)
    , flags(0) {
}

SceNgsBufferInfo *ModuleData::lock_params(const MemState &mem) {
    const std::lock_guard<std::mutex> guard(*parent->voice_mutex);

    // Save a copy of previous set of data
    if (flags & PARAMS_LOCK) {
        return nullptr;
    }

    const uint8_t *current_data = info.data.cast<const uint8_t>().get(mem);
    last_info.resize(info.size);
    memcpy(last_info.data(), current_data, info.size);

    flags |= PARAMS_LOCK;

    return &info;
}

bool ModuleData::unlock_params(const MemState &mem) {
    const std::lock_guard<std::mutex> guard(*parent->voice_mutex);

    parent->rack->modules[index]->on_param_change(mem, *this);

    if (flags & PARAMS_LOCK) {
        flags &= ~PARAMS_LOCK;
        return true;
    }

    return false;
}

void ModuleData::invoke_callback(KernelState &kernel, const MemState &mem, const SceUID thread_id, const uint32_t reason1,
    const uint32_t reason2, Address reason_ptr) {
    return parent->invoke_callback(kernel, mem, thread_id, callback, user_data, parent->rack->modules[index]->module_id(),
        reason1, reason2, reason_ptr);
}

void Voice::init(Rack *mama) {
    rack = mama;
    state = VoiceState::VOICE_STATE_AVAILABLE;
    is_pending = false;
    is_paused = false;
    is_keyed_off = false;

    datas.resize(mama->modules.size());

    for (uint32_t i = 0; i < MAX_OUTPUT_PORT; i++)
        patches[i].resize(mama->patches_per_output);

    inputs.init(rack->system->granularity, 1);
    voice_mutex = std::make_unique<std::mutex>();
}

Ptr<Patch> Voice::patch(const MemState &mem, const int32_t index, int32_t subindex, int32_t dest_index, Ptr<Voice> source, Ptr<Voice> dest) {
    const std::lock_guard<std::mutex> guard(*voice_mutex);

    if (index >= MAX_OUTPUT_PORT) {
        // We don't have enough port for you!
        return {};
    }

    // Look if another patch has already been there
    if (subindex == -1) {
        for (int32_t i = 0; i < patches[index].size(); i++) {
            if (!patches[index][i] || (patches[index])[i].get(mem)->output_sub_index == -1) {
                subindex = i;
                break;
            }
        }
    }

    if (subindex >= patches[index].size()) {
        return {};
    }

    if (patches[index][subindex] && patches[index][subindex].get(mem)->output_sub_index != -1) {
        // You just hit an occupied subindex! You won't get to eat, stay in detention.
        return {};
    }

    if (!patches[index][subindex]) {
        // Create the patch incase it's not yet existed
        patches[index][subindex] = rack->alloc_and_init<Patch>(mem);
    }

    Patch *patch = patches[index][subindex].get(mem);

    patch->output_sub_index = subindex;
    patch->output_index = index;
    patch->dest_index = dest_index;
    patch->dest = dest;
    patch->source = source;

    // Initialize the matrix
    memset(patch->volume_matrix, 0, sizeof(patch->volume_matrix));

    return patches[index][subindex];
}

bool Voice::remove_patch(const MemState &mem, const Ptr<Patch> patch) {
    if (!patch || !voice_mutex) {
        return false;
    }
    const std::lock_guard<std::mutex> guard(*voice_mutex);
    bool found = false;
    for (auto &patches_1 : patches) {
        if (std::ranges::contains(patches_1, patch)) {
            found = true;
            break;
        }
    }
    if (!found)
        return false;

    // Try to unroute. Free the destination index
    patch.get(mem)->output_sub_index = -1;

    return true;
}

ModuleData *Voice::module_storage(const uint32_t index) {
    if (index >= datas.size()) {
        return nullptr;
    }

    return &datas[index];
}

void Voice::transition(const MemState &mem, const VoiceState new_state) {
    const VoiceState old = state;
    state = new_state;

    for (size_t i = 0; i < datas.size(); i++) {
        rack->modules[i]->on_state_change(mem, datas[i], old);
    }
}

bool Voice::parse_params(const MemState &mem, const SceNgsModuleParamHeader *header) {
    ModuleData *storage = module_storage(header->module_id);

    if (!storage)
        return false;

    if (storage->flags & ModuleData::PARAMS_LOCK)
        return false;

    const auto *descr = reinterpret_cast<const SceNgsParamsDescriptor *>(header + 1);
    if (descr->size > storage->info.size)
        return false;

    memcpy(storage->info.data.get(mem), descr, descr->size);

    return true;
}

SceInt32 Voice::parse_params_block(const MemState &mem, const SceNgsModuleParamHeader *header, const SceUInt32 size) {
    const SceUInt8 *data = reinterpret_cast<const SceUInt8 *>(header);
    const SceUInt8 *data_end = data + size;

    SceInt32 num_error = 0;

    // after first loop, check if other module exist
    while (data < data_end) {
        if (!parse_params(mem, header))
            num_error++;

        // increment by the size of the header alone + the descriptor size
        data += sizeof(SceNgsModuleParamHeader) + reinterpret_cast<const SceNgsParamsDescriptor *>(header + 1)->size;

        // set new header for next module
        header = reinterpret_cast<const SceNgsModuleParamHeader *>(data);
    }

    return num_error;
}

bool Voice::set_preset(const MemState &mem, const SceNgsVoicePreset *preset) {
    // we ignore the name for now
    const uint8_t *data_origin = reinterpret_cast<const uint8_t *>(preset);

    if (preset->preset_data_offset) {
        const auto *preset_data = reinterpret_cast<const SceNgsModuleParamHeader *>(data_origin + preset->preset_data_offset);
        auto nb_errors = parse_params_block(mem, preset_data, preset->preset_data_size);
        if (nb_errors > 0)
            return false;
    }

    if (preset->bypass_flags_offset) {
        const auto *bypass_flags = reinterpret_cast<const SceUInt32 *>(data_origin + preset->bypass_flags_offset);
        // should we disable bypass on all modules first?
        for (SceUInt32 i = 0; i < preset->bypass_flags_nb; i++) {
            ModuleData *module_data = module_storage(*bypass_flags);
            if (!module_data)
                return false;
            module_data->is_bypassed = true;

            bypass_flags++;
        }
    }

    return true;
}

void Voice::invoke_callback(KernelState &kernel, const MemState &mem, const SceUID thread_id, Ptr<void> callback, Ptr<void> user_data,
    const uint32_t module_id, const uint32_t reason1, const uint32_t reason2, Address reason_ptr) {
    if (!callback) {
        return;
    }

    const ThreadStatePtr thread = kernel.get_thread(thread_id);
    const Address callback_info_addr = stack_alloc(*thread->cpu, sizeof(SceNgsCallbackInfo));

    SceNgsCallbackInfo *info = Ptr<SceNgsCallbackInfo>(callback_info_addr).get(mem);
    info->rack_handle = Ptr<void>(rack, mem);
    info->voice_handle = Ptr<void>(this, mem);
    info->module_id = module_id;
    info->callback_reason = reason1;
    info->callback_reason_2 = reason2;
    info->callback_ptr = Ptr<void>(reason_ptr);
    info->userdata = user_data;

    thread->run_callback(callback.address(), { callback_info_addr });
    stack_free(*thread->cpu, sizeof(SceNgsCallbackInfo));
}

uint32_t System::get_required_memspace_size(SceNgsSystemInitParams *parameters) {
    return sizeof(System);
}

uint32_t Rack::get_required_memspace_size(MemState &mem, SceNgsRackDescription *description) {
    uint32_t buffer_size = 0;
    if (description->definition)
        // multiply by 2 because there are 2 copies of each buffer
        buffer_size = 2 * get_voice_definition_size(description->definition.get(mem));

    return sizeof(ngs::Rack) + description->voice_count * (sizeof(ngs::Voice) + buffer_size + description->patches_per_output * description->definition.get(mem)->output_count * sizeof(ngs::Patch));
}

bool init(State &ngs, MemState &mem) {
    voice_definition_init(ngs, mem);

    return true;
}

void deinit(State &ngs, MemState &mem) {
    while (!ngs.systems.empty()) {
        release_system(ngs, mem, ngs.systems.back());
    }

    Atrac9Module::free_swr_contexts();

    ngs.definitions = Ptr<VoiceDefinition>(0);
}

void collect_host_owned_ranges(State &ngs, const MemState &mem,
    std::vector<std::pair<Address, uint32_t>> &out) {
    // Must be called *before* guest memory is overwritten -- this walk is only safe while the
    // host pointers embedded in these objects are still this process's own.
    //
    // Only the objects themselves, not the mempool they sit in. The pool also holds the guest's
    // own parameter buffers, and those have to be rewound with everything else: leaving the whole
    // pool alone meant the host voice state described a moment the restored guest knew nothing
    // about, and sceNgsSystemUpdate walked into it and aborted on a bogus vector length.
    for (System *system : ngs.systems) {
        if (!system)
            continue;
        out.emplace_back(Ptr<System>(system, mem).address(), static_cast<uint32_t>(sizeof(System)));
        for (Rack *rack : system->racks) {
            if (!rack)
                continue;
            out.emplace_back(Ptr<Rack>(rack, mem).address(), static_cast<uint32_t>(sizeof(Rack)));
            for (const Ptr<Voice> &voice : rack->voices) {
                if (voice)
                    out.emplace_back(voice.address(), static_cast<uint32_t>(sizeof(Voice)));
            }
        }
    }
}

// --- moving NGS objects to where a savestate had them ---------------------------------------------
//
// System, Rack and Voice are host objects constructed into memory the game handed over: a system
// at the start of its own pool, each rack at the start of its pool, and a rack's voices, their
// parameter buffers and their patches allocated inside the rack's pool in a fixed order. The
// game's own heap decides where those pools are, and a session that took a different path can put
// them somewhere else. After a load the guest holds the state's addresses, so the live objects
// have to be there. When they were not, sceNgsSystemUpdate was handed a system nobody knew, the
// mix was skipped, and the music became one buffer played over and over.
//
// A pool moves as a unit: everything inside keeps its offset and every guest pointer into it
// shifts by the same amount. The objects themselves are carried as bytes. On this toolchain
// (MSVC, release, no iterator debugging) std::vector, std::deque, an unlocked recursive_mutex and
// a condition_variable_any with no waiters hold no pointers to themselves, so a byte copy is a
// move. The scheduler must be idle (scheduler_busy), and a guest thread waiting on its condition
// variable never reaches the pause barrier, so no load gets this far with one.

SavedLayout capture_layout(State &ngs, const MemState &mem) {
    SavedLayout layout;
    layout.definitions = ngs.definitions.address();
    for (System *system : ngs.systems) {
        SavedLayout::SystemEntry entry;
        if (system) {
            entry.addr = Ptr<System>(system, mem).address();
            for (Rack *rack : system->racks) {
                SavedLayout::RackEntry rack_entry;
                if (rack) {
                    rack_entry.addr = Ptr<Rack>(rack, mem).address();
                    for (const Ptr<Voice> &voice : rack->voices)
                        rack_entry.voices.push_back(voice.address());
                }
                entry.racks.push_back(std::move(rack_entry));
            }
        }
        layout.systems.push_back(std::move(entry));
    }
    return layout;
}

// The extent of a pool, from its allocator. At least `minimum`, so the object at its start is
// always inside it.
static uint32_t pool_extent(const MempoolObject &pool, uint32_t minimum) {
    uint32_t end = minimum;
    for (const auto &block : pool.allocator.blocks)
        end = std::max(end, block.offset + block.size);
    return end;
}

std::string check_relocatable(State &ngs, const MemState &mem, const SavedLayout &saved) {
    if (saved.definitions != ngs.definitions.address())
        return fmt::format("voice definitions are at {} here but {} in the state",
            log_hex(ngs.definitions.address()), log_hex(saved.definitions));
    if (saved.systems.size() != ngs.systems.size())
        return fmt::format("{} system(s) here, {} in the state", ngs.systems.size(), saved.systems.size());

    for (size_t i = 0; i < saved.systems.size(); i++) {
        const System *system = ngs.systems[i];
        const SavedLayout::SystemEntry &saved_system = saved.systems[i];
        if (!system != !saved_system.addr)
            return fmt::format("system {} exists in one session and not the other", i);
        if (!system)
            continue;
        if (saved_system.racks.size() != system->racks.size())
            return fmt::format("system {} has {} rack slot(s) here, {} in the state", i, system->racks.size(), saved_system.racks.size());

        for (size_t j = 0; j < saved_system.racks.size(); j++) {
            const Rack *rack = system->racks[j];
            const SavedLayout::RackEntry &saved_rack = saved_system.racks[j];
            if (!rack != !saved_rack.addr)
                return fmt::format("rack {} of system {} exists in one session and not the other", j, i);
            if (!rack)
                continue;
            if (saved_rack.voices.size() != rack->voices.size())
                return fmt::format("rack {} has {} voice(s) here, {} in the state", j, rack->voices.size(), saved_rack.voices.size());

            // A pool can only move as a unit if everything in it is where it was relative to the
            // start. Allocation inside a rack is deterministic, so this should always hold; if it
            // does not, moving would put voices where the guest does not expect them.
            const Address live_rack = Ptr<Rack>(const_cast<Rack *>(rack), mem).address();
            for (size_t k = 0; k < saved_rack.voices.size(); k++) {
                if (saved_rack.voices[k] - saved_rack.addr != rack->voices[k].address() - live_rack)
                    return fmt::format("voice {} of rack {} sits at a different place in its pool", k, j);
            }
        }
    }
    return {};
}

bool scheduler_busy(State &ngs) {
    for (System *system : ngs.systems) {
        if (!system)
            continue;
        std::unique_lock<std::recursive_mutex> lock(system->voice_scheduler.mutex, std::try_to_lock);
        if (!lock.owns_lock() || system->voice_scheduler.is_updating)
            return true;
    }
    return false;
}

RelocationStash stash_for_relocation(State &ngs, const MemState &mem, const SavedLayout &saved) {
    RelocationStash stash;
    const uint8_t *const base = mem.memory.get();
    const auto carry = [&](Address from, Address to, uint32_t size) {
        stash.blocks.push_back({ from, to, std::vector<uint8_t>(base + from, base + from + size) });
    };

    for (size_t i = 0; i < saved.systems.size(); i++) {
        System *system = ngs.systems[i];
        if (!system)
            continue;
        const SavedLayout::SystemEntry &saved_system = saved.systems[i];
        const Address live_system = Ptr<System>(system, mem).address();
        stash.moves.push_back({ live_system, pool_extent(*system, sizeof(System)), saved_system.addr });
        carry(live_system, saved_system.addr, sizeof(System));

        for (size_t j = 0; j < saved_system.racks.size(); j++) {
            Rack *rack = system->racks[j];
            if (!rack)
                continue;
            const SavedLayout::RackEntry &saved_rack = saved_system.racks[j];
            const Address live_rack = Ptr<Rack>(rack, mem).address();
            stash.moves.push_back({ live_rack, pool_extent(*rack, sizeof(Rack)), saved_rack.addr });
            carry(live_rack, saved_rack.addr, sizeof(Rack));
            for (size_t k = 0; k < rack->voices.size(); k++) {
                if (rack->voices[k])
                    carry(rack->voices[k].address(), saved_rack.voices[k], sizeof(Voice));
            }
        }
    }
    return stash;
}

uint32_t relocate_after_restore(State &ngs, MemState &mem, const RelocationStash &stash) {
    uint8_t *const base = mem.memory.get();
    for (const RelocationStash::Block &block : stash.blocks)
        std::memcpy(base + block.to, block.bytes.data(), block.bytes.size());

    // A guest address inside a pool that moved, moved with it. Anything else stays.
    const auto map_guest = [&](Address addr) -> Address {
        for (const RelocationStash::Move &move : stash.moves) {
            if (addr >= move.from && addr < move.from + move.size)
                return addr - move.from + move.to;
        }
        return addr;
    };
    const auto map_host = [&](auto *pointer) -> decltype(pointer) {
        if (!pointer)
            return pointer;
        const Address addr = static_cast<Address>(reinterpret_cast<uint8_t *>(pointer) - base);
        return reinterpret_cast<decltype(pointer)>(base + map_guest(addr));
    };
    const auto map_ptr = [&](auto &ptr) {
        if (ptr)
            ptr = std::remove_reference_t<decltype(ptr)>(map_guest(ptr.address()));
    };

    for (System *&system : ngs.systems) {
        if (!system)
            continue;
        system = map_host(system);
        map_ptr(system->memspace);
        for (Rack *&rack : system->racks)
            rack = map_host(rack);
        for (Voice *&voice : system->voice_scheduler.queue)
            voice = map_host(voice);

        std::queue<OperationPending> pending;
        while (!system->voice_scheduler.operations_pending.empty()) {
            OperationPending op = system->voice_scheduler.operations_pending.front();
            system->voice_scheduler.operations_pending.pop();
            op.system = map_host(op.system);
            if (op.type == PendingType::ReleaseRack)
                op.release_data.rack = map_host(op.release_data.rack);
            pending.push(op);
        }
        system->voice_scheduler.operations_pending = std::move(pending);

        for (Rack *rack : system->racks) {
            if (!rack)
                continue;
            map_ptr(rack->memspace);
            rack->system = map_host(rack->system);
            for (Ptr<Voice> &voice_ptr : rack->voices) {
                map_ptr(voice_ptr);
                Voice *voice = voice_ptr.get(mem);
                if (!voice)
                    continue;
                voice->rack = map_host(voice->rack);
                for (auto &port : voice->patches) {
                    for (Ptr<Patch> &patch : port)
                        map_ptr(patch);
                }
                map_ptr(voice->finished_callback);
                map_ptr(voice->finished_callback_user_data);
                // Filled afresh at the start of every update.
                std::memset(voice->products, 0, sizeof(voice->products));
                for (ModuleData &data : voice->datas) {
                    data.parent = voice;
                    map_ptr(data.info.data);
                    map_ptr(data.callback);
                    map_ptr(data.user_data);
                }
            }
        }
    }

    return static_cast<uint32_t>(std::count_if(stash.moves.begin(), stash.moves.end(),
        [](const RelocationStash::Move &move) { return move.from != move.to; }));
}

std::vector<SavedVoice> capture_voices(State &ngs, const MemState &mem) {
    std::vector<SavedVoice> out;
    for (System *system : ngs.systems) {
        if (!system)
            continue;
        for (Rack *rack : system->racks) {
            if (!rack)
                continue;
            for (const Ptr<Voice> &voice_ptr : rack->voices) {
                Voice *voice = voice_ptr.get(mem);
                if (!voice)
                    continue;
                SavedVoice saved;
                saved.addr = voice_ptr.address();
                saved.state = voice->state;
                saved.is_pending = voice->is_pending;
                saved.is_paused = voice->is_paused;
                saved.is_keyed_off = voice->is_keyed_off;
                saved.frame_count = voice->frame_count;
                for (const ModuleData &data : voice->datas) {
                    SavedVoice::Module module;
                    module.guest_state = data.guest_state_data;
                    module.loop_count = data.logical_state ? data.logical_state->loop_count() : 0;
                    saved.modules.push_back(std::move(module));
                }
                out.push_back(std::move(saved));
            }
        }
    }
    return out;
}

uint32_t restore_voices(State &ngs, const MemState &mem, const std::vector<SavedVoice> &saved) {
    std::map<Address, const SavedVoice *> by_address;
    for (const SavedVoice &voice : saved)
        by_address[voice.addr] = &voice;

    uint32_t restored = 0;
    for (System *system : ngs.systems) {
        if (!system)
            continue;
        const std::lock_guard<std::recursive_mutex> guard(system->voice_scheduler.mutex);
        for (Rack *rack : system->racks) {
            if (!rack)
                continue;
            for (const Ptr<Voice> &voice_ptr : rack->voices) {
                Voice *voice = voice_ptr.get(mem);
                const auto it = by_address.find(voice_ptr.address());
                if (!voice || it == by_address.end() || it->second->modules.size() != voice->datas.size())
                    continue;
                const SavedVoice &from = *it->second;
                voice->state = static_cast<VoiceState>(from.state);
                voice->is_pending = from.is_pending;
                voice->is_paused = from.is_paused;
                voice->is_keyed_off = from.is_keyed_off;
                voice->frame_count = from.frame_count;
                for (size_t i = 0; i < voice->datas.size(); i++) {
                    ModuleData &data = voice->datas[i];
                    data.guest_state_data = from.modules[i].guest_state;
                    if (data.logical_state)
                        data.logical_state->restart_at_position(from.modules[i].loop_count);
                }
                const bool playing = voice->state == VOICE_STATE_ACTIVE || voice->state == VOICE_STATE_FINALIZING;
                system->voice_scheduler.set_queued(mem, voice, playing && !voice->is_paused);
                restored++;
            }
        }
    }
    return restored;
}

void on_savestate_loaded(State &ngs, const MemState &mem) {
    for (System *system : ngs.systems) {
        if (!system)
            continue;

        // Keep the scheduler exactly as it is.
        //
        // The queue holds the voices that are playing, and those Voice objects are preserved across
        // the load (see collect_host_owned_ranges), so the queue still describes them correctly.
        // Clearing it was tried and was wrong: the game starts its music and master voices once,
        // when a scene begins, and never queues them again. After a load the scheduler sat at zero
        // voices, the guest kept submitting whatever was left in its output buffer, and the music
        // became one short fragment looping -- on every load, measured as 2 voices queued before
        // and 0 after, for good.
        for (Rack *rack : system->racks) {
            if (!rack)
                continue;
            for (const Ptr<Voice> &voice_ptr : rack->voices) {
                Voice *voice = voice_ptr.get(mem);
                if (!voice)
                    continue;
                for (ModuleData &data : voice->datas) {
                    if (data.runtime_state)
                        data.runtime_state->on_savestate_loaded();
                }
            }
        }
    }
}

bool init_system(State &ngs, const MemState &mem, SceNgsSystemInitParams *parameters, Ptr<void> memspace, const uint32_t memspace_size) {
    // Reserve first memory allocation for our System struct
    System *sys = memspace.cast<System>().get(mem);
    sys = new (sys) System(memspace, memspace_size);

    sys->racks.resize(parameters->max_racks);

    sys->max_voices = parameters->max_voices;
    sys->granularity = parameters->granularity;
    sys->sample_rate = parameters->sample_rate;

    // Alloc first block for System struct
    if (!sys->alloc_raw(sizeof(System))) {
        return false;
    }

    ngs.systems.push_back(sys);
    return true;
}

void release_system(State &ngs, const MemState &mem, System *system) {
    // this function assumes no ngs mutex is being held
    for (Rack *rack : system->racks) {
        if (!rack)
            continue;
        for (const auto &voice : rack->voices) {
            system->voice_scheduler.deque_voice(voice.get(mem));
            voice.get(mem)->~Voice();
        }
        rack->~Rack();
    }

    system->racks.clear();

    vector_utils::erase_first(ngs.systems, system);
    system->~System();
}

bool init_rack(State &ngs, const MemState &mem, System *system, SceNgsBufferInfo *init_info, const SceNgsRackDescription *description) {
    Rack *rack = init_info->data.cast<Rack>().get(mem);
    rack = new (rack) Rack(system, init_info->data, init_info->size);

    // Alloc first block for Rack
    if (!rack->alloc<Rack>()) {
        return false;
    }

    if (description->definition)
        apply_voice_definition(description->definition.get(mem), rack->modules);
    else
        rack->modules.clear();

    // Initialize voice definition
    rack->channels_per_voice = description->channels_per_voice;
    rack->max_patches_per_input = description->max_patches_per_input;
    rack->patches_per_output = description->patches_per_output;

    // Alloc spaces for voice
    rack->voices.resize(description->voice_count);
    rack->vdef = description->definition.get(mem);

    for (auto &voice : rack->voices) {
        voice = rack->alloc<Voice>();

        if (!voice) {
            return false;
        }

        Voice *v = voice.get(mem);
        new (v) Voice();
        v->init(rack);

        // Allocate parameter buffer info for each voice
        for (size_t i = 0; i < rack->modules.size(); i++) {
            v->datas[i].info.size = rack->modules[i]->get_buffer_parameter_size();
            v->datas[i].info.data = rack->alloc_raw(v->datas[i].info.size);
            // from the behavior of games, it looks like the other info buffer (there are two copies because of VoiceLock) is located right after the first
            // one, so copy this behavior, to avoid a game overwriting some important ngs struct
            rack->alloc_raw(v->datas[i].info.size);

            v->datas[i].parent = v;
            v->datas[i].index = static_cast<uint32_t>(i);
            rack->modules[i]->initialize_voice_data(v->datas[i]);
        }
    }

    system->racks.push_back(rack);

    return true;
}

void release_rack(State &ngs, const MemState &mem, System *system, Rack *rack) {
    // this function should only be called outside of ngs update and with the scheduler mutex acquired (except when releasing the system)
    if (!rack)
        return;

    // remove all queued voices
    for (const auto &voice : rack->voices) {
        Voice *v = voice.get(mem);
        system->voice_scheduler.deque_voice(voice.get(mem));
        // clean up host resources per voice before destroying
        for (size_t i = 0; i < rack->modules.size() && i < v->datas.size(); i++) {
            if (rack->modules[i])
                rack->modules[i]->cleanup_voice_state(v->datas[i]);
        }
        // no need to free the voice from the rack
        v->~Voice();
    }

    // remove from system
    vector_utils::erase_first(system->racks, rack);

    // free pointer memory
    rack->~Rack();
}

Ptr<VoiceDefinition> get_voice_definition(State &ngs, MemState &mem, ngs::BussType type) {
    return ngs.definitions + static_cast<int>(type);
}
} // namespace ngs
