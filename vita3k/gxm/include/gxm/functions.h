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

#include <gxm/state.h>
#include <gxm/types.h>

#include <array>
#include <bitset>
#include <string>
#include <utility>
#include <vector>

struct EmuEnvState;
struct GxmState;

namespace gxm {
// Guest byte ranges that hold host objects.
//
// SceGxmSyncObject carries a std::mutex and a condition_variable; SceGxmContext and
// SceGxmRenderTarget carry std::unique_ptrs to renderer objects. All three are constructed into
// *guest* memory, so their bytes end up in a savestate, and restoring a state taken by a
// different process installs that process's host pointers. Callers leave these ranges alone.
// Collect before guest memory is overwritten. See ngs::collect_host_owned_memspaces for the
// same problem in NGS.
void collect_host_owned_ranges(GxmState &gxm, const MemState &mem,
    std::vector<std::pair<Address, uint32_t>> &out);

// Where every GXM host object was when a savestate was taken, by kind, each list sorted.
//
// These objects are not moved by a load (unlike the NGS ones), so a state is only safe to load if
// every object it refers to is still at the same address in this session. When the game's heap is
// laid out differently they are not: measured by pushing the heap up 16 MiB before loading a state
// from a normal session, which moved the immediate context and the shader patcher by exactly that,
// and the first sceGxmBeginScene after the load followed the saved context's dead renderer pointer.
struct HostObjectLayout {
    std::vector<Address> contexts;
    std::vector<Address> render_targets;
    std::vector<Address> sync_objects;
    std::vector<Address> shader_patchers;
    std::vector<Address> vertex_programs;
    std::vector<Address> fragment_programs;
};

HostObjectLayout capture_layout(GxmState &gxm, const MemState &mem);

// Empty if every object in `saved` is a live object of the same kind at the same address here,
// otherwise which one is not. Extra live objects are not fine -- see rollback_to_layout.
std::string check_layout(GxmState &gxm, const MemState &mem, const HostObjectLayout &saved);

// Destroy the host half of every object that is live here but not in `saved`: the ones the guest
// created after the state was taken. Their guest memory came from the guest's own heap (shader
// patchers, programs) or from the allocator a load reconciles (sync objects), and a load rewinds
// both to a moment when that memory was free. Keeping the host half alive across that is what
// made a quickload of Oddworld die a few seconds later: the rewound guest reused the bytes of a
// vertex program it had made after the save, the patcher's cache still handed that program out on
// the next key hit, and the draw read its renderer_data as guest data (measured: refcount
// 2409865712, renderer_data 0x3f800000 -- the float 1.0 -- program 0x1; then a fault in
// gxmSetUniformBuffers at 0x290000002C, or a hang). Programs are taken out of their patcher's
// cache and destructed in place, patchers likewise, sync objects dropped from the registry; the
// bytes are left for the restore to overwrite with what the guest had there. Nothing is handed
// back to the guest heap: the load is about to put that heap back as it was.
//
// Render targets and contexts are destroyed through the render thread, which is parked for the
// whole load, so a state that has extras of those kinds is refused (non-empty return, nothing
// touched) rather than left half-handled. Counts of what was destroyed come back in `out`.
struct RollbackCounts {
    uint32_t vertex_programs = 0;
    uint32_t fragment_programs = 0;
    uint32_t shader_patchers = 0;
    uint32_t sync_objects = 0;
    uint32_t total() const {
        return vertex_programs + fragment_programs + shader_patchers + sync_objects;
    }
};
std::string rollback_to_layout(GxmState &gxm, const MemState &mem, const HostObjectLayout &saved, RollbackCounts &out);

// Whether the guest is between sceGxmBeginScene and sceGxmEndScene on its immediate context. The
// renderer mirrors this once it has run everything submitted, so a savestate taken or loaded at
// such a moment leaves the renderer recording a scene the restored guest knows nothing about.
bool scene_in_progress(GxmState &gxm, const MemState &mem);

// Color.
SceGxmColorBaseFormat get_base_format(SceGxmColorFormat src);
size_t bits_per_pixel(SceGxmColorBaseFormat base_format);
size_t get_stride_in_bytes(const SceGxmColorFormat src, const std::size_t stride_in_pixels);

// Textures.
uint32_t get_width(const SceGxmTexture &texture);
uint32_t get_height(const SceGxmTexture &texture);
SceGxmTextureFormat get_format(const SceGxmTexture &texture);
SceGxmTextureBaseFormat get_base_format(SceGxmTextureFormat src);
uint32_t get_num_components(SceGxmTextureBaseFormat fmt);
std::pair<uint32_t, uint32_t> get_block_size(SceGxmTextureBaseFormat base_format);
uint32_t get_stride_in_bytes(const SceGxmTexture &texture);
uint32_t bits_per_pixel(SceGxmTextureBaseFormat base_format);
// get the size of the first mip of the first face
uint32_t texture_size_first_mip(const SceGxmTexture &texture);
bool is_bcn_format(SceGxmTextureBaseFormat base_format);
bool is_pvrt_format(SceGxmTextureBaseFormat base_format);
bool is_block_compressed_format(SceGxmTextureBaseFormat base_format);
bool is_paletted_format(SceGxmTextureBaseFormat base_format);
bool is_yuv_format(SceGxmTextureBaseFormat base_format);
uint32_t attribute_format_size(SceGxmAttributeFormat format);
bool is_stream_instancing(SceGxmIndexSource source);
bool convert_color_format_to_texture_format(SceGxmColorFormat format, SceGxmTextureFormat &dest_format);

// Transfer
uint32_t get_bits_per_pixel(SceGxmTransferFormat Format);

void destroy_all_contexts(EmuEnvState &emuenv, bool force_backend_destroy);
void destroy_all_render_targets(EmuEnvState &emuenv, bool force_backend_destroy);
void shutdown(EmuEnvState &emuenv);
void invalidate_sync_objects(GxmState &gxm);
} // namespace gxm

namespace gxp {
// Used to map GXM program parameters to GLSL data types
enum class GenericParameterType {
    Scalar,
    Vector,
    Matrix,
    Array
};

using GxmVertexOutputTexCoordInfos = std::array<uint8_t, 10>;

void log_parameter(const SceGxmProgramParameter &parameter);

/**
 * \brief If parameter belongs in a struct, returns the struct field name only
 */
std::string parameter_name(const SceGxmProgramParameter &parameter);

/**
 * \brief If parameter belongs in a struct, returns the struct name only
 */
std::string parameter_struct_name(const SceGxmProgramParameter &parameter);
GenericParameterType parameter_generic_type(const SceGxmProgramParameter &parameter);
/**
 * \return SceGxmVertexProgramOutput (bitfield)
 */
SceGxmVertexProgramOutputs get_vertex_outputs(const SceGxmProgram &program, GxmVertexOutputTexCoordInfos *coord_infos = nullptr);
SceGxmFragmentProgramInputs get_fragment_inputs(const SceGxmProgram &program);

int get_parameter_type_size(const SceGxmParameterType type);
int get_num_32_bit_components(const SceGxmParameterType type, const uint16_t num_comp);

const SceGxmProgramParameterContainer *get_container_by_index(const SceGxmProgram &program, const std::uint16_t idx);
const char *get_container_name(const std::uint16_t idx);

int get_uniform_buffer_base(const SceGxmProgram &program, const SceGxmProgramParameter &parameter);

typedef std::bitset<SCE_GXM_MAX_TEXTURE_UNITS> TextureInfo;
TextureInfo get_textures_used(const SceGxmProgram &program_gxp);

} // namespace gxp
