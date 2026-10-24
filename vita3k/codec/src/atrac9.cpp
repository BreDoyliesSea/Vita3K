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

#include <codec/state.h>

extern "C" {
#include <libatrac9.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <structures.h>
}

#include <error_codes.h>

#include <util/log.h>

#include <algorithm>
#include <cstring>

struct FFMPEGAtrac9Info {
    uint32_t version;
    uint32_t config_data;
    uint32_t padding;
};

uint32_t Atrac9DecoderState::get(DecoderQuery query) {
    Atrac9CodecInfo *info = static_cast<Atrac9CodecInfo *>(atrac9_info);

    switch (query) {
    case DecoderQuery::CHANNELS: return info->channels;
    // The bit rate is the size of a superframe times the number of superframes per second (times 8)
    case DecoderQuery::BIT_RATE: return static_cast<uint32_t>((info->superframeSize * 8ULL * info->samplingRate) / (info->frameSamples * info->framesInSuperframe));
    case DecoderQuery::SAMPLE_RATE: return info->samplingRate;
    case DecoderQuery::AT9_SAMPLE_PER_FRAME: return info->frameSamples;
    case DecoderQuery::AT9_SAMPLE_PER_SUPERFRAME: return info->frameSamples * info->framesInSuperframe;
    case DecoderQuery::AT9_FRAMES_IN_SUPERFRAME: return info->framesInSuperframe;
    case DecoderQuery::AT9_SUPERFRAME_SIZE: return info->superframeSize;
    default: return 0;
    }
}

uint32_t Atrac9DecoderState::get_es_size() {
    return es_size_used;
}

void Atrac9DecoderState::flush() {
    Atrac9CodecInfo *info = static_cast<Atrac9CodecInfo *>(atrac9_info);
    superframe_frame_idx = 0;
    superframe_data_left = info->superframeSize;

    Frame &frame = static_cast<Atrac9Handle *>(decoder_handle)->Frame;
    frame.IndexInSuperframe = 0;
    if (frame.Channels[0])
        std::fill_n(frame.Channels[0]->Mdct.ImdctPrevious, 256, 0.0);
    if (frame.Channels[1])
        std::fill_n(frame.Channels[1]->Mdct.ImdctPrevious, 256, 0.0);
}

void Atrac9DecoderState::export_state(Atrac9DecoderSavedState *dest) {
    Frame &frame = static_cast<Atrac9Handle *>(decoder_handle)->Frame;
    if (frame.Channels[0])
        std::copy_n(frame.Channels[0]->Mdct.ImdctPrevious, 256, dest->prev_values[0]);
    if (frame.Channels[1])
        std::copy_n(frame.Channels[1]->Mdct.ImdctPrevious, 256, dest->prev_values[1]);
}

void Atrac9DecoderState::load_state(const Atrac9DecoderSavedState *src) {
    Frame &frame = static_cast<Atrac9Handle *>(decoder_handle)->Frame;
    if (frame.Channels[0])
        std::copy_n(src->prev_values[0], 256, frame.Channels[0]->Mdct.ImdctPrevious);
    if (frame.Channels[1])
        std::copy_n(src->prev_values[1], 256, frame.Channels[1]->Mdct.ImdctPrevious);
}

namespace {
// The blob: this header, then every Block of the frame verbatim. A Block holds all a frame hands
// to the next -- scale factors, band parameters, the IMDCT overlap, the noise generator -- plus
// pointers into its own decoder, which restore_position puts back from the live one.
struct Atrac9PositionHeader {
    uint32_t block_size;
    uint32_t config_data;
    int32_t superframe_frame_idx;
    int32_t superframe_data_left;
    int32_t index_in_superframe;
};
} // namespace

std::vector<uint8_t> Atrac9DecoderState::save_position() const {
    const Atrac9Handle *const handle = static_cast<Atrac9Handle *>(decoder_handle);
    const Atrac9PositionHeader header{ static_cast<uint32_t>(sizeof(Block)), config_data, superframe_frame_idx,
        superframe_data_left, handle->Frame.IndexInSuperframe };
    std::vector<uint8_t> out(sizeof(header) + sizeof(handle->Frame.Blocks));
    std::memcpy(out.data(), &header, sizeof(header));
    std::memcpy(out.data() + sizeof(header), handle->Frame.Blocks, sizeof(handle->Frame.Blocks));
    return out;
}

bool Atrac9DecoderState::restore_position(const std::vector<uint8_t> &saved) {
    Atrac9Handle *const handle = static_cast<Atrac9Handle *>(decoder_handle);
    Frame &frame = handle->Frame;
    Atrac9PositionHeader header;
    if (saved.size() != sizeof(header) + sizeof(frame.Blocks))
        return false;
    std::memcpy(&header, saved.data(), sizeof(header));
    if (header.block_size != sizeof(Block) || header.config_data != config_data)
        return false;

    for (int b = 0; b < MAX_BLOCK_COUNT; b++) {
        Block &block = frame.Blocks[b];
        const Frame *const block_frame = block.Frame;
        const ConfigData *const block_config = block.Config;
        struct {
            Frame *frame;
            Block *block;
            ConfigData *config;
            double *window;
            double *sin_table;
            double *cos_table;
        } live[MAX_BLOCK_CHANNEL_COUNT];
        for (int c = 0; c < MAX_BLOCK_CHANNEL_COUNT; c++) {
            const Channel &channel = block.Channels[c];
            live[c] = { channel.Frame, channel.Block, channel.Config, channel.Mdct.Window, channel.Mdct.SinTable, channel.Mdct.CosTable };
        }

        std::memcpy(&block, saved.data() + sizeof(header) + b * sizeof(Block), sizeof(Block));

        block.Frame = const_cast<Frame *>(block_frame);
        block.Config = const_cast<ConfigData *>(block_config);
        for (int c = 0; c < MAX_BLOCK_CHANNEL_COUNT; c++) {
            Channel &channel = block.Channels[c];
            channel.Frame = live[c].frame;
            channel.Block = live[c].block;
            channel.Config = live[c].config;
            channel.Mdct.Window = live[c].window;
            channel.Mdct.SinTable = live[c].sin_table;
            channel.Mdct.CosTable = live[c].cos_table;
        }
    }
    frame.IndexInSuperframe = header.index_in_superframe;
    superframe_frame_idx = header.superframe_frame_idx;
    superframe_data_left = header.superframe_data_left;
    return true;
}

bool Atrac9DecoderState::send(const uint8_t *data, uint32_t size) {
    Atrac9CodecInfo *info = static_cast<Atrac9CodecInfo *>(atrac9_info);

    int decode_used = 0;

    const int res = Atrac9Decode(decoder_handle, data, reinterpret_cast<short *>(result.data()), &decode_used);
    if (res != At9Status::ERR_SUCCESS) {
        LOG_ERROR("Decode failure with code {}", log_hex(res));
        return false;
    }

    es_size_used = static_cast<uint32_t>(decode_used);
    superframe_data_left -= decode_used;
    superframe_frame_idx++;
    if (superframe_frame_idx == info->framesInSuperframe) {
        // add the padding between two superframes as size used if there is
        es_size_used += superframe_data_left;
        superframe_frame_idx = 0;
        superframe_data_left = info->superframeSize;
    }

    return true;
}

bool Atrac9DecoderState::receive(uint8_t *data, DecoderSize *size) {
    Atrac9CodecInfo *info = static_cast<Atrac9CodecInfo *>(atrac9_info);

    if (data) {
        memcpy(data, result.data(), info->frameSamples * info->channels * sizeof(uint16_t));
    }

    if (size) {
        size->samples = static_cast<uint32_t>(info->frameSamples);
    }

    return true;
}

Atrac9DecoderState::Atrac9DecoderState(uint32_t config_data)
    : config_data(config_data) {
    decoder_handle = Atrac9GetHandle();
    const int err = Atrac9InitDecoder(decoder_handle, reinterpret_cast<uint8_t *>(&config_data));

    if (err != At9Status::ERR_SUCCESS) {
        LOG_ERROR("Error initializing decoder. Error code: {}", log_hex(err));
    }

    Atrac9CodecInfo *info = new Atrac9CodecInfo;
    atrac9_info = info;
    Atrac9GetCodecInfo(decoder_handle, info);

    result.resize(info->frameSamples * info->channels * sizeof(uint16_t));

    superframe_frame_idx = 0;
    superframe_data_left = info->superframeSize;
}

Atrac9DecoderState::~Atrac9DecoderState() {
    Atrac9ReleaseHandle(decoder_handle);
    delete static_cast<Atrac9CodecInfo *>(atrac9_info);
    context = nullptr;
}
