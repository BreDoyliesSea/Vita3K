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

#include "SceAudiodecUser.h"

#include <audio/state.h>
#include <codec/state.h>
#include <kernel/state.h>
#include <util/lock_and_find.h>
#include <util/tracy.h>

#include <cstring>

TRACY_MODULE_NAME(SceAudiodecUser);

enum {
    SCE_AUDIODEC_ERROR_API_FAIL = 0x807F0000,
    SCE_AUDIODEC_ERROR_INVALID_TYPE = 0x807F0001,
    SCE_AUDIODEC_ERROR_NOT_INITIALIZED = 0x807F0005,
    SCE_AUDIODEC_ERROR_INVALID_PTR = 0x807F0008,
    SCE_AUDIODEC_ERROR_INVALID_HANDLE = 0x807F0009,
    SCE_AUDIODEC_ERROR_NOT_HANDLE_IN_USE = 0x807F000A,
    SCE_AUDIODEC_ERROR_INVALID_SIZE = 0x807F000D,
    SCE_AUDIODEC_AT9_ERROR_INVALID_CONFIG = 0x807F2000,
    SCE_AUDIODEC_MP3_ERROR_INVALID_MPEG_VERSION = 0x807F2801,
};

enum {
    SCE_AUDIODEC_MP3_MPEG_VERSION_2_5,
    SCE_AUDIODEC_MP3_MPEG_VERSION_RESERVED,
    SCE_AUDIODEC_MP3_MPEG_VERSION_2,
    SCE_AUDIODEC_MP3_MPEG_VERSION_1,
};

typedef std::shared_ptr<DecoderState> DecoderPtr;
typedef std::map<SceUID, DecoderPtr> DecoderStates;
typedef std::set<SceUID> CodecDecoders;
typedef std::map<SceAudiodecCodec, CodecDecoders> CodecDecodersMap;

struct AudiodecState {
    std::mutex mutex;
    DecoderStates decoders;
    CodecDecodersMap codecs;
};

struct SceAudiodecInfoAt9 {
    uint32_t config_data;
    uint32_t channels;
    uint32_t bit_rate;
    uint32_t sample_rate;
    uint32_t super_frame_size;
    uint32_t frames_in_super_frame;
};

struct SceAudiodecInfoMp3 {
    uint32_t channels;
    uint32_t version;
};

struct SceAudiodecInfoAac {
    uint32_t is_adts;
    uint32_t channels;
    uint32_t sample_rate;
    uint32_t is_sbr;
};

struct SceAudiodecInfoCelp {
    uint32_t excitation_mode;
    uint32_t sample_rate;
    uint32_t bit_rate;
    uint32_t lost_count;
};

struct SceAudiodecInfo {
    uint32_t size;
    union {
        SceAudiodecInfoAt9 at9;
        SceAudiodecInfoMp3 mp3;
        SceAudiodecInfoAac aac;
        SceAudiodecInfoCelp celp;
    };
};

struct SceAudiodecCtrl {
    uint32_t size;
    SceUID handle;
    Ptr<uint8_t> es_data;
    uint32_t es_size_used;
    uint32_t es_size_max;
    Ptr<uint8_t> pcm_data;
    uint32_t pcm_size_given;
    uint32_t pcm_size_max;
    uint32_t word_length;
    Ptr<SceAudiodecInfo> info;
};

static_assert(sizeof(SceAudiodecCtrl) == 0x28);

constexpr uint32_t SCE_AUDIODEC_AT9_MAX_ES_SIZE = 1024;
constexpr uint32_t SCE_AUDIODEC_MP3_MAX_ES_SIZE = 1441;
// max size is 1792 for AAC ES if adts is enabled
constexpr uint32_t SCE_AUDIODEC_AAC_MAX_ES_SIZE = 1536;
constexpr uint32_t SCE_AUDIODEC_CELP_MAX_ES_SIZE = 27;

// this value is multiplied by 2 if sbr is enabled
constexpr uint32_t SCE_AUDIODEC_AAC_MAX_PCM_SIZE = KiB(2);
constexpr uint32_t SCE_AUDIODEC_MP3_V1_MAX_PCM_SIZE = 2304;
constexpr uint32_t SCE_AUDIODEC_MP3_V2_MAX_PCM_SIZE = 1152;

LIBRARY_INIT(SceAudiodec) {
    emuenv.kernel.obj_store.create<AudiodecState>();
}

// Declared in codec/state.h. A load rewinds guest memory, and with it every stream the guest feeds
// these decoders from, while each decoder keeps its place in the stream it was decoding. Flush
// them all to a clean start; audiodec_restore_positions then puts the ATRAC9 ones back where the
// state had them, which a clean start is not (see there).
size_t audiodec_flush_after_savestate_load(EmuEnvState &emuenv) {
    // Always present while a game runs: LIBRARY_INIT creates it for every title at startup, and
    // only the kernel's deinit at exit clears it.
    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    const std::lock_guard<std::mutex> lock(state->mutex);
    for (const auto &[_, decoder] : state->decoders) {
        if (decoder)
            decoder->flush();
    }
    return state->decoders.size();
}

// Declared in codec/state.h. A flush is not enough for ATRAC9: the decoder counts the frames of the
// current superframe and carries data from each frame to the next, and the restored guest is
// usually part-way through a superframe. Flushed, the decoder took the guest's next frame for the
// first of a superframe and refused the one after it, which builds on its predecessor -- measured
// on Gravity Rush (run 41): the guest resumed at exactly the right byte, the second frame after the
// load failed with ERR_UNPACK_SCALE_FACTOR_MODE_INVALID, and the music never came back. Without
// the flush the live decoder's later position was just as wrong. So a save records each ATRAC9
// decoder's position by handle.
std::vector<uint8_t> audiodec_save_positions(EmuEnvState &emuenv) {
    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    const std::lock_guard<std::mutex> lock(state->mutex);
    std::vector<uint8_t> out(sizeof(uint32_t));
    const auto append = [&out](const void *data, size_t size) {
        const auto *const bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    uint32_t count = 0;
    for (const auto &[handle, decoder] : state->decoders) {
        const auto *const at9 = dynamic_cast<const Atrac9DecoderState *>(decoder.get());
        if (!at9)
            continue;
        const std::vector<uint8_t> position = at9->save_position();
        const auto length = static_cast<uint32_t>(position.size());
        append(&handle, sizeof(handle));
        append(&length, sizeof(length));
        append(position.data(), position.size());
        count++;
    }
    std::memcpy(out.data(), &count, sizeof(count));
    return out;
}

// Declared in codec/state.h. Runs after audiodec_flush_after_savestate_load. A decoder the state has
// no position for, or one opened with a different configuration, stays flushed.
size_t audiodec_restore_positions(EmuEnvState &emuenv, const uint8_t *data, size_t size) {
    if (!data)
        return 0;
    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    const std::lock_guard<std::mutex> lock(state->mutex);
    size_t at = 0;
    const auto read = [&](void *dest, size_t length) {
        if (size - at < length)
            return false;
        std::memcpy(dest, data + at, length);
        at += length;
        return true;
    };
    uint32_t count = 0;
    if (!read(&count, sizeof(count)))
        return 0;
    size_t restored = 0;
    for (uint32_t i = 0; i < count; i++) {
        SceUID handle = 0;
        uint32_t length = 0;
        if (!read(&handle, sizeof(handle)) || !read(&length, sizeof(length)) || size - at < length)
            break;
        const std::vector<uint8_t> position(data + at, data + at + length);
        at += length;
        const auto it = state->decoders.find(handle);
        auto *const at9 = it != state->decoders.end() ? dynamic_cast<Atrac9DecoderState *>(it->second.get()) : nullptr;
        if (at9 && at9->restore_position(position))
            restored++;
    }
    return restored;
}

// Declared in codec/state.h. The codec each handle was opened for, and what its decoder was made
// with. Kept small on purpose: everything else about a decoder is either in guest memory or in the
// position blob ADEC already carries.
std::vector<uint8_t> audiodec_save_identities(EmuEnvState &emuenv) {
    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    const std::lock_guard<std::mutex> lock(state->mutex);
    std::vector<uint8_t> out(sizeof(uint32_t));
    const auto append = [&out](const void *data, size_t size) {
        const auto *const bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    uint32_t count = 0;
    for (const auto &[handle, decoder] : state->decoders) {
        if (!decoder)
            continue;
        uint32_t codec = UINT32_MAX;
        for (const auto &[codec_id, handles] : state->codecs) {
            if (handles.contains(handle)) {
                codec = static_cast<uint32_t>(codec_id);
                break;
            }
        }
        uint32_t first = 0, second = 0;
        if (const auto *const at9 = dynamic_cast<const Atrac9DecoderState *>(decoder.get())) {
            first = at9->config_data;
        } else if (const auto *const aac = dynamic_cast<const AacDecoderState *>(decoder.get())) {
            first = aac->sample_rate;
            second = aac->channels;
        } else {
            // Nothing recorded means nothing recreated; the load says so rather than guessing.
            codec = UINT32_MAX;
        }
        append(&handle, sizeof(handle));
        append(&codec, sizeof(codec));
        append(&first, sizeof(first));
        append(&second, sizeof(second));
        count++;
    }
    std::memcpy(out.data(), &count, sizeof(count));
    return out;
}

// Declared in codec/state.h. Runs before the flush and the position restore, so a decoder created
// here is then put where the state had it like any other.
size_t audiodec_recreate_missing(EmuEnvState &emuenv, const uint8_t *data, size_t size) {
    if (!data)
        return 0;
    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    const std::lock_guard<std::mutex> lock(state->mutex);
    size_t at = 0;
    const auto read = [&](void *dest, size_t length) {
        if (size - at < length)
            return false;
        std::memcpy(dest, data + at, length);
        at += length;
        return true;
    };
    uint32_t count = 0;
    if (!read(&count, sizeof(count)))
        return 0;
    size_t recreated = 0;
    for (uint32_t i = 0; i < count; i++) {
        SceUID handle = 0;
        uint32_t codec = 0, first = 0, second = 0;
        if (!read(&handle, sizeof(handle)) || !read(&codec, sizeof(codec)) || !read(&first, sizeof(first)) || !read(&second, sizeof(second)))
            break;
        if (state->decoders.contains(handle))
            continue;
        DecoderPtr decoder;
        switch (codec) {
        case SCE_AUDIODEC_TYPE_AT9:
            decoder = std::make_shared<Atrac9DecoderState>(first);
            break;
        case SCE_AUDIODEC_TYPE_AAC:
            decoder = std::make_shared<AacDecoderState>(first, second);
            break;
        default:
            LOG_WARN("Savestate: decoder {} in the state was opened for codec {}, which cannot be created again; the guest will be told the handle is invalid",
                handle, codec == UINT32_MAX ? std::string("an unrecorded one") : fmt::format("{}", codec));
            continue;
        }
        state->decoders[handle] = decoder;
        state->codecs[static_cast<SceAudiodecCodec>(codec)].insert(handle);
        // So this session does not hand the same number out again.
        emuenv.kernel.ensure_next_uid_above(handle);
        recreated++;
    }
    return recreated;
}

EXPORT(int, sceAudiodecClearContext, SceAudiodecCtrl *ctrl) {
    TRACY_FUNC(sceAudiodecClearContext, ctrl)

    if (!ctrl)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    if (ctrl->size != sizeof(SceAudiodecCtrl))
        return SCE_AUDIODEC_ERROR_INVALID_SIZE;

    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    if (state->codecs.empty()) {
        return SCE_AUDIODEC_ERROR_NOT_INITIALIZED;
    }
    if (!ctrl->handle) {
        return SCE_AUDIODEC_ERROR_INVALID_HANDLE;
    }

    const DecoderPtr &decoder = lock_and_find(ctrl->handle, state->decoders, state->mutex);

    if (decoder) {
        decoder->flush();
    } else {
        return SCE_AUDIODEC_ERROR_NOT_HANDLE_IN_USE;
    }

    return 0;
}

static int create_decoder(EmuEnvState &emuenv, SceAudiodecCtrl *ctrl, SceAudiodecCodec codec) {
    if (!ctrl)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    if (ctrl->size != sizeof(SceAudiodecCtrl))
        return SCE_AUDIODEC_ERROR_INVALID_SIZE;

    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    std::lock_guard<std::mutex> lock(state->mutex);

    SceUID handle = emuenv.kernel.get_next_uid();
    ctrl->handle = handle;
    state->codecs[codec].insert(handle);

    switch (codec) {
    case SCE_AUDIODEC_TYPE_AT9: {
        SceAudiodecInfoAt9 &info = ctrl->info.get(emuenv.mem)->at9;
        DecoderPtr decoder = std::make_shared<Atrac9DecoderState>(info.config_data);
        state->decoders[handle] = decoder;

        info.channels = decoder->get(DecoderQuery::CHANNELS);
        info.bit_rate = decoder->get(DecoderQuery::BIT_RATE);
        info.sample_rate = decoder->get(DecoderQuery::SAMPLE_RATE);
        info.super_frame_size = decoder->get(DecoderQuery::AT9_SUPERFRAME_SIZE);
        info.frames_in_super_frame = decoder->get(DecoderQuery::AT9_FRAMES_IN_SUPERFRAME);
        ctrl->es_size_max = std::min(info.super_frame_size, SCE_AUDIODEC_AT9_MAX_ES_SIZE);
        ctrl->pcm_size_max = decoder->get(DecoderQuery::AT9_SAMPLE_PER_FRAME)
            * decoder->get(DecoderQuery::CHANNELS) * sizeof(int16_t);
        return 0;
    }
    case SCE_AUDIODEC_TYPE_AAC: {
        SceAudiodecInfoAac &info = ctrl->info.get(emuenv.mem)->aac;
        DecoderPtr decoder = std::make_shared<AacDecoderState>(info.sample_rate, info.channels);
        state->decoders[handle] = decoder;

        ctrl->es_size_max = SCE_AUDIODEC_AAC_MAX_ES_SIZE;
        if (info.is_adts)
            ctrl->es_size_max += 0x100;
        ctrl->pcm_size_max = info.channels * SCE_AUDIODEC_AAC_MAX_PCM_SIZE;
        if (info.is_sbr)
            ctrl->pcm_size_max *= 2;

        LOG_WARN_IF(info.is_adts || info.is_sbr, "report it to dev, is_adts: {}, is_sbr: {}", info.is_adts, info.is_sbr);

        return 0;
    }
    case SCE_AUDIODEC_TYPE_MP3: {
        SceAudiodecInfoMp3 &info = ctrl->info.get(emuenv.mem)->mp3;
        DecoderPtr decoder = std::make_shared<Mp3DecoderState>(info.channels);
        state->decoders[handle] = decoder;

        ctrl->es_size_max = SCE_AUDIODEC_MP3_MAX_ES_SIZE;

        switch (info.version) {
        case SCE_AUDIODEC_MP3_MPEG_VERSION_1:
            ctrl->pcm_size_max = info.channels * SCE_AUDIODEC_MP3_V1_MAX_PCM_SIZE;
            return 0;
        case SCE_AUDIODEC_MP3_MPEG_VERSION_2:
        case SCE_AUDIODEC_MP3_MPEG_VERSION_2_5:
            ctrl->pcm_size_max = info.channels * SCE_AUDIODEC_MP3_V2_MAX_PCM_SIZE;
            return 0;
        default:
            LOG_ERROR("Invalid MPEG version {}.", info.version);
            return SCE_AUDIODEC_MP3_ERROR_INVALID_MPEG_VERSION;
        }
    }
    default: {
        LOG_ERROR("Unimplemented audio decoder {}.", codec);
        return -1;
    }
    }
}

EXPORT(int, sceAudiodecCreateDecoder, SceAudiodecCtrl *ctrl, SceAudiodecCodec codec) {
    TRACY_FUNC(sceAudiodecCreateDecoder, ctrl, codec);
    return create_decoder(emuenv, ctrl, codec);
}

EXPORT(int, sceAudiodecCreateDecoderExternal, SceAudiodecCtrl *ctrl, SceAudiodecCodec codec, void *context, uint32_t size) {
    TRACY_FUNC(sceAudiodecCreateDecoderExternal, ctrl, codec, context, size);
    // I think context is supposed to be just extra memory where I can allocate my context.
    // I'm just going to allocate like regular sceAudiodecCreateDecoder and see how it goes.
    // Almost sure zang has already tried this so :/ - desgroup
    return create_decoder(emuenv, ctrl, codec);
}

EXPORT(int, sceAudiodecCreateDecoderResident) {
    TRACY_FUNC(sceAudiodecCreateDecoderResident);
    return UNIMPLEMENTED();
}

static int decode_audio_frames(EmuEnvState &emuenv, const char *export_name, SceAudiodecCtrl *ctrl, SceUInt32 nb_frames) {
    if (!ctrl)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    if (ctrl->size != sizeof(SceAudiodecCtrl))
        return SCE_AUDIODEC_ERROR_INVALID_SIZE;

    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    const DecoderPtr &decoder = lock_and_find(ctrl->handle, state->decoders, state->mutex);
    // A handle with no decoder behind it used to be dereferenced anyway, which is a host crash on
    // the guest's thread rather than an error the guest can see. A savestate load can produce one
    // (see audiodec_recreate_missing), and so can a guest that decodes on a deleted handle.
    if (!decoder)
        return RET_ERROR(SCE_AUDIODEC_ERROR_INVALID_HANDLE);

    uint8_t *es_data = ctrl->es_data.get(emuenv.mem);
    uint8_t *pcm_data = ctrl->pcm_data.get(emuenv.mem);

    ctrl->es_size_used = 0;
    ctrl->pcm_size_given = 0;

    for (uint32_t frame = 0; frame < nb_frames; frame++) {
        DecoderSize size;
        if (!decoder->send(es_data, ctrl->es_size_max)
            || !decoder->receive(pcm_data, &size)) {
            // Flush before reporting the failure. A decoder keeps its position within the current
            // superframe across calls, so a frame that fails to unpack leaves that position
            // part-way into data it can no longer make sense of, and every subsequent frame fails
            // the same way -- one bad frame becomes permanent silence.
            //
            // The NGS path already does exactly this ("flush or we'll get an error next time we
            // want to decode" in ngs/src/modules/atrac9.cpp). This one did not, which is why audio
            // never recovered here. The frame really did fail, so the error is still returned;
            // this only stops it being contagious.
            decoder->flush();
            return RET_ERROR(SCE_AUDIODEC_ERROR_API_FAIL);
        }

        uint32_t es_size_used = std::min(decoder->get_es_size(), ctrl->es_size_max);
        assert(es_size_used <= ctrl->es_size_max);
        ctrl->es_size_used += es_size_used;
        es_data += es_size_used;

        uint32_t pcm_size_given = size.samples * decoder->get(DecoderQuery::CHANNELS) * sizeof(int16_t);
        assert(pcm_size_given <= ctrl->pcm_size_max);
        ctrl->pcm_size_given += pcm_size_given;
        pcm_data += pcm_size_given;
    }

    return 0;
}

EXPORT(int, sceAudiodecDecode, SceAudiodecCtrl *ctrl) {
    TRACY_FUNC(sceAudiodecDecode, ctrl);
    return decode_audio_frames(emuenv, export_name, ctrl, 1);
}

EXPORT(int, sceAudiodecDecodeNFrames, SceAudiodecCtrl *ctrl, SceUInt32 nFrames) {
    TRACY_FUNC(sceAudiodecDecodeNFrames, ctrl, nFrames);
    return decode_audio_frames(emuenv, export_name, ctrl, nFrames);
}

EXPORT(int, sceAudiodecDecodeNStreams, SceAudiodecCtrl *ctrl, SceUInt32 nStreams) {
    TRACY_FUNC(sceAudiodecDecodeNStreams, ctrl, nStreams);

    if (!ctrl)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    if (ctrl->size != sizeof(SceAudiodecCtrl))
        return SCE_AUDIODEC_ERROR_INVALID_SIZE;

    return UNIMPLEMENTED();
}

EXPORT(int, sceAudiodecDeleteDecoder, SceAudiodecCtrl *ctrl) {
    TRACY_FUNC(sceAudiodecDeleteDecoder, ctrl);

    if (!ctrl)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    if (ctrl->size != sizeof(SceAudiodecCtrl))
        return SCE_AUDIODEC_ERROR_INVALID_SIZE;

    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    std::lock_guard<std::mutex> lock(state->mutex);
    state->decoders.erase(ctrl->handle);

    // there are at most 4 different codecs, we can afford to look
    // at all of them (the handle is in one of them)
    for (auto &codec : state->codecs) {
        codec.second.erase(ctrl->handle);
    }

    return 0;
}

EXPORT(int, sceAudiodecDeleteDecoderExternal, SceAudiodecCtrl *ctrl, void *context) {
    TRACY_FUNC(sceAudiodecDeleteDecoderExternal, ctrl, context);
    return CALL_EXPORT(sceAudiodecDeleteDecoder, ctrl);
}

EXPORT(int, sceAudiodecDeleteDecoderResident) {
    TRACY_FUNC(sceAudiodecDeleteDecoderResident);
    return UNIMPLEMENTED();
}

static std::uint32_t getAt9Factor(const std::uint8_t *config_data) {
    std::uint32_t value = (config_data[1] & 0xf) >> 1;
    if (value == 0)
        return 1;
    if ((value == 1) || (value == 2))
        return 2;
    return SCE_AUDIODEC_AT9_ERROR_INVALID_CONFIG;
}

EXPORT(int, sceAudiodecGetContextSize, SceAudiodecCtrl *pCtrl, SceAudiodecCodec codecType) {
    TRACY_FUNC(sceAudiodecGetContextSize, pCtrl, codecType);

    if (!pCtrl)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    if (pCtrl->size != sizeof(SceAudiodecCtrl))
        return SCE_AUDIODEC_ERROR_INVALID_SIZE;

    switch (codecType) {
    case SCE_AUDIODEC_TYPE_AT9: {
        const std::uint32_t at9Factor = getAt9Factor(reinterpret_cast<std::uint8_t *>(&pCtrl->info.get(emuenv.mem)->at9.config_data));
        if (at9Factor == 1 || at9Factor == 2) {
            return 0x400 * at9Factor + 0x400;
        }
        return SCE_AUDIODEC_AT9_ERROR_INVALID_CONFIG;
    }
    case SCE_AUDIODEC_TYPE_AAC:
        return 0x18000;
    case SCE_AUDIODEC_TYPE_MP3:
    case SCE_AUDIODEC_TYPE_CELP:
        return 0;
    default:
        // Found these during reverse engineering, log them in case we need an implementation
        LOG_WARN_IF(codecType == 0x1007 || codecType == 0x1008, "Unsupported codec type {}", codecType);
        return SCE_AUDIODEC_ERROR_INVALID_TYPE;
    }
}

EXPORT(int, sceAudiodecGetInternalError) {
    TRACY_FUNC(sceAudiodecGetInternalError);
    return UNIMPLEMENTED();
}

EXPORT(SceInt32, sceAudiodecInitLibrary, SceAudiodecCodec codecType, SceAudiodecInitParam *pInitParam) {
    TRACY_FUNC(sceAudiodecInitLibrary, codecType, pInitParam);

    if (!pInitParam)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    std::lock_guard<std::mutex> lock(state->mutex);

    state->codecs[codecType] = CodecDecoders();
    return 0;
}

EXPORT(int, sceAudiodecPartlyDecode, SceAudiodecCtrl *ctrl, SceUInt32 samples_offset, SceUInt32 samples_to_decode) {
    TRACY_FUNC(sceAudiodecPartlyDecode, ctrl, samples_offset, samples_to_decode);

    if (!ctrl)
        return SCE_AUDIODEC_ERROR_INVALID_PTR;

    if (ctrl->size != sizeof(SceAudiodecCtrl))
        return SCE_AUDIODEC_ERROR_INVALID_SIZE;

    // this function is only called by libatrac
    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    if (!state->codecs[SCE_AUDIODEC_TYPE_AT9].contains(ctrl->handle)) {
        STUBBED("Call to sceAudiodecPartlyDecode with a codec other than Atrac9, report it to the devs");
    }

    const std::shared_ptr<DecoderState> &decoder = lock_and_find(ctrl->handle, state->decoders, state->mutex);

    uint8_t *es_data = ctrl->es_data.get(emuenv.mem);
    uint8_t *pcm_data = ctrl->pcm_data.get(emuenv.mem);

    // TODO: if the offset is too big, do not decode the first superframes (doesn't seem to happen with libatrac)
    const uint32_t bytes_per_sample = decoder->get(DecoderQuery::CHANNELS) * sizeof(int16_t);
    ctrl->es_size_used = 0;
    ctrl->pcm_size_given = 0;
    std::vector<uint8_t> temp_storage;
    temp_storage.reserve((samples_offset + samples_to_decode) * bytes_per_sample);

    while (ctrl->pcm_size_given < (samples_offset + samples_to_decode) * bytes_per_sample) {
        DecoderSize size;
        if (!decoder->send(es_data, ctrl->es_size_max)) {
            return RET_ERROR(SCE_AUDIODEC_ERROR_API_FAIL);
        }
        const uint32_t es_size_used = decoder->get_es_size();
        assert(es_size_used <= ctrl->es_size_max);
        ctrl->es_size_used += es_size_used;
        es_data += es_size_used;

        decoder->receive(nullptr, &size);
        const uint32_t pcm_size_given = size.samples * bytes_per_sample;
        ctrl->pcm_size_given += pcm_size_given;
        const uint32_t old_size = temp_storage.size();
        temp_storage.resize(old_size + pcm_size_given);
        decoder->receive(temp_storage.data() + old_size, &size);
    }

    memcpy(pcm_data + samples_offset * bytes_per_sample, temp_storage.data() + samples_offset * bytes_per_sample, samples_to_decode * bytes_per_sample);

    return 0;
}

EXPORT(SceInt32, sceAudiodecTermLibrary, SceAudiodecCodec codecType) {
    TRACY_FUNC(sceAudiodecTermLibrary, codecType);
    const auto state = emuenv.kernel.obj_store.get<AudiodecState>();
    std::lock_guard<std::mutex> lock(state->mutex);

    // remove decoders associated with codecType
    for (auto &handle : state->codecs[codecType]) {
        state->decoders.erase(handle);
    }
    state->codecs.erase(codecType);
    return 0;
}
