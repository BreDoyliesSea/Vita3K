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

#include "SceSysmem.h"
#include "SceSysmemForDriver.h"

#include <emuenv/state.h>
#include <kernel/state.h>
#include <kernel/types.h>
#include <modules/sysmem_state.h>

#include <cstring>

#include <packages/sfo.h>

#include <util/align.h>
#include <util/string_utils.h>

#include <util/tracy.h>
TRACY_MODULE_NAME(SceSysmem);

std::string to_debug_str(const MemState &mem, SceKernelMemBlockType type) {
    switch (type) {
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE: return "SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE";
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_RX: return "SCE_KERNEL_MEMBLOCK_TYPE_USER_RX";
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_RW: return "SCE_KERNEL_MEMBLOCK_TYPE_USER_RW";
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW: return "SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW";
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW: return "SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW";
    case SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW: return "SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW";
    }
    return std::to_string(type);
}

struct SceKernelAllocMemBlockOpt {
    SceSize size;
    SceUInt32 attr;
    SceSize alignment;
    SceUInt32 uidBaseBlock;
    const char *strBaseBlockName;
    int flags;
    int reserved[10];
};

struct SceKernelFreeMemorySizeInfo {
    int size; //!< sizeof(SceKernelFreeMemorySizeInfo)
    int size_user; //!< Free memory size for *_USER_RW memory
    int size_cdram; //!< Free memory size for USER_CDRAM_RW memory
    int size_phycont; //!< Free memory size for USER_MAIN_PHYCONT_*_RW memory
};

LIBRARY_INIT(SceSysmem) {
    emuenv.kernel.obj_store.create<SysmemState>();
}

// Declared in modules/sysmem_state.h. The registry only: uid, name, where the block is and how big,
// plus which of them the VM variants own and what the running totals were.
std::vector<uint8_t> sysmem_save_blocks(EmuEnvState &emuenv) {
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const std::lock_guard<std::mutex> lock(state->mutex);
    std::vector<uint8_t> out;
    const auto append = [&out](const void *data, size_t size) {
        const auto *const bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    const auto append32 = [&append](uint32_t value) { append(&value, sizeof(value)); };

    append32(static_cast<uint32_t>(state->blocks.size()));
    for (const auto &[uid, block] : state->blocks) {
        if (!block)
            continue;
        append(&uid, sizeof(uid));
        const auto name_length = static_cast<uint32_t>(std::strlen(block->name));
        append32(name_length);
        append(block->name, name_length);
        append32(block->size);
        append32(block->mappedBase.address());
        append32(block->mappedSize);
        append32(static_cast<uint32_t>(block->memoryType));
        append32(block->access);
        append32(static_cast<uint32_t>(block->type));
        append32(state->vm_blocks.contains(uid) ? 1 : 0);
    }
    append32(static_cast<uint32_t>(state->next_uid));
    append32(state->allocated_user);
    append32(state->allocated_cdram);
    append32(state->allocated_phycont);
    return out;
}

// Declared in modules/sysmem_state.h. Runs after the memory pass, which has already put the pages
// back; this only makes the kernel own up to them again. Blocks this session has that the state does
// not are left alone, like the allocations behind them: the restored guest cannot see them, and
// freeing one would pull the ground out from under something host-side still pointing at it.
size_t sysmem_recreate_missing_blocks(EmuEnvState &emuenv, const uint8_t *data, size_t size) {
    if (!data)
        return 0;
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const std::lock_guard<std::mutex> lock(state->mutex);
    size_t at = 0;
    const auto read = [&](void *dest, size_t length) {
        if (size - at < length)
            return false;
        std::memcpy(dest, data + at, length);
        at += length;
        return true;
    };
    const auto read32 = [&](uint32_t &value) { return read(&value, sizeof(value)); };

    uint32_t count = 0;
    if (!read32(count))
        return 0;
    size_t recreated = 0;
    for (uint32_t i = 0; i < count; i++) {
        SceUID uid = 0;
        uint32_t name_length = 0;
        if (!read(&uid, sizeof(uid)) || !read32(name_length) || size - at < name_length)
            return recreated;
        const std::string name(reinterpret_cast<const char *>(data + at), name_length);
        at += name_length;
        uint32_t info_size = 0, mapped_base = 0, mapped_size = 0, memory_type = 0, access = 0, type = 0, is_vm = 0;
        if (!read32(info_size) || !read32(mapped_base) || !read32(mapped_size) || !read32(memory_type)
            || !read32(access) || !read32(type) || !read32(is_vm))
            return recreated;
        if (state->blocks.contains(uid))
            continue;
        const KernelMemBlockPtr block = std::make_shared<KernelMemBlock>();
        block->size = info_size;
        block->mappedBase = Ptr<void>(mapped_base);
        block->mappedSize = mapped_size;
        block->memoryType = static_cast<int>(memory_type);
        block->access = access;
        block->type = static_cast<SceKernelMemBlockType>(type);
        std::strncpy(block->name, name.c_str(), KERNELOBJECT_MAX_NAME_LENGTH);
        block->name[KERNELOBJECT_MAX_NAME_LENGTH] = '\0';
        state->blocks.emplace(uid, block);
        if (is_vm)
            state->vm_blocks.emplace(uid, block);
        recreated++;
    }
    uint32_t next_uid = 0, allocated_user = 0, allocated_cdram = 0, allocated_phycont = 0;
    if (read32(next_uid) && read32(allocated_user) && read32(allocated_cdram) && read32(allocated_phycont)) {
        // So this session does not hand a recreated block's number out again.
        state->next_uid = std::max<SceUID>(state->next_uid, static_cast<SceUID>(next_uid));
        state->allocated_user = allocated_user;
        state->allocated_cdram = allocated_cdram;
        state->allocated_phycont = allocated_phycont;
    }
    return recreated;
}

EXPORT(SceUID, sceKernelAllocMemBlock, const char *pName, SceKernelMemBlockType type, SceSize size, SceKernelAllocMemBlockOpt *optp) {
    TRACY_FUNC(sceKernelAllocMemBlock, pName, type, size, optp);

    // Build kernel opts from user opts
    SceKernelAllocMemBlockKernelOpt k_opt = {};
    k_opt.size = sizeof(k_opt);
    SceKernelAllocMemBlockKernelOpt *k_opt_ptr = nullptr;

    if (optp) {
        if (optp->attr & SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_HAS_ALIGNMENT) {
            k_opt.attr |= SCE_KERNEL_ALLOC_MEMBLOCK_ATTR_HAS_ALIGNMENT;
            k_opt.alignment = optp->alignment;
        }
        k_opt_ptr = &k_opt;
    }

    return CALL_EXPORT(ksceKernelAllocMemBlock, pName, type, size, k_opt_ptr);
}

EXPORT(int, sceKernelAllocMemBlockForVM, const char *pName, SceSize size) {
    TRACY_FUNC(sceKernelAllocMemBlockForVM, pName, size);
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const auto guard = std::lock_guard<std::mutex>(state->mutex);

    MemState &mem = emuenv.mem;
    assert(pName != nullptr);

    if (size < 0x1000 || (size & 0xFFF) != 0) {
        return RET_ERROR(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }

    const Ptr<void> address(alloc(mem, size, pName));
    if (!address) {
        return RET_ERROR(SCE_KERNEL_ERROR_NO_MEMORY);
    }

    const SceUID uid = state->get_next_uid();

    const KernelMemBlockPtr sceKernelMemBlock = std::make_shared<KernelMemBlock>();
    sceKernelMemBlock->mappedBase = address;
    sceKernelMemBlock->mappedSize = size;
    sceKernelMemBlock->size = sizeof(SceKernelMemBlockInfo);
    std::strncpy(sceKernelMemBlock->name, pName, KERNELOBJECT_MAX_NAME_LENGTH);
    state->blocks.emplace(uid, sceKernelMemBlock);
    state->vm_blocks.emplace(uid, sceKernelMemBlock);
    state->allocated_user += size;

    return uid;
}

EXPORT(int, sceKernelAllocUnmapMemBlock) {
    TRACY_FUNC(sceKernelAllocUnmapMemBlock);
    return UNIMPLEMENTED();
}

EXPORT(int, sceKernelCheckModelCapability) {
    TRACY_FUNC(sceKernelCheckModelCapability);
    return UNIMPLEMENTED();
}

EXPORT(int, sceKernelCloseMemBlock) {
    TRACY_FUNC(sceKernelCloseMemBlock);
    return UNIMPLEMENTED();
}

EXPORT(int, sceKernelCloseVMDomain) {
    TRACY_FUNC(sceKernelCloseVMDomain);
    return UNIMPLEMENTED();
}

EXPORT(SceUID, sceKernelFindMemBlockByAddr, Address addr, uint32_t size) {
    TRACY_FUNC(sceKernelFindMemBlockByAddr, addr, size);
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const auto guard = std::lock_guard<std::mutex>(state->mutex);

    for (auto &[id, block] : state->blocks) {
        if (block->mappedBase.address() <= addr && (block->mappedBase.address() + block->mappedSize > addr)) {
            return id;
        }
    }
    return RET_ERROR(SCE_KERNEL_ERROR_BLOCK_ERROR);
}

EXPORT(int, sceKernelFreeMemBlock, SceUID uid) {
    TRACY_FUNC(sceKernelFreeMemBlock, uid);
    return CALL_EXPORT(ksceKernelFreeMemBlock, uid);
}

EXPORT(int, sceKernelFreeMemBlockForVM, SceUID uid) {
    TRACY_FUNC(sceKernelFreeMemBlockForVM, uid);
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const auto guard = std::lock_guard<std::mutex>(state->mutex);

    assert(uid >= 0);
    const Blocks::const_iterator block = state->vm_blocks.find(uid);
    assert(block != state->vm_blocks.end());

    free(emuenv.mem, block->second->mappedBase.address());
    state->allocated_user -= block->second->size;
    state->blocks.erase(block);
    state->vm_blocks.erase(block);

    return SCE_KERNEL_OK;
}

EXPORT(int, sceKernelGetFreeMemorySize, SceKernelFreeMemorySizeInfo *info) {
    TRACY_FUNC(sceKernelGetFreeMemorySize, info);

    // Default memory configuration
    uint32_t max_user = MiB(256);

    // if DevKit then max_user = MB(512); else check sfo file for memory expansion mode
    // Fetch the "ATTRIBUTE2" key from the SFO file to check for memory expansion mode
    std::string attribute2;
    if (sfo::get_data_by_key(attribute2, emuenv.sfo_handle, "ATTRIBUTE2")) {
        // Convert the string to an unsigned 32-bit integer
        const uint32_t attr_val = string_utils::stoi_def(attribute2, 0, "memory expansion mode");

        switch (attr_val & 0x0C) {
        case 0x4: // Check for the +29MiB mode
            max_user += MiB(29);
            break;
        case 0x8: // Check for the +77MiB mode
            max_user += MiB(77);
            break;
        case 0xC: // Check for the +109MiB mode
            max_user += MiB(109);
            break;
        default: break;
        }
    } else
        LOG_WARN_ONCE("ATTRIBUTE2 key not found in SFO data.");

    // Define other memory limits
    constexpr uint32_t max_cdram = MiB(112); // Max cdram memory (112 MiB)
    constexpr uint32_t max_phycont = MiB(26); // Max physically contiguous memory (26 MiB)
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const auto guard = std::lock_guard<std::mutex>(state->mutex);

    // Set the free memory size info
    info->size_cdram = std::max<int>(max_cdram - state->allocated_cdram, 0);
    info->size_user = std::max<int>(max_user - state->allocated_user, 0);
    info->size_phycont = std::max<int>(max_phycont - state->allocated_phycont, 0);

    return 0;
}

EXPORT(int, sceKernelGetMemBlockBase, SceUID uid, Ptr<void> *basep) {
    TRACY_FUNC(sceKernelGetMemBlockBase, uid, basep);
    return CALL_EXPORT(ksceKernelGetMemBlockBase, uid, basep);
}

EXPORT(int, sceKernelGetMemBlockInfoByAddr, Address addr, SceKernelMemBlockInfo *info) {
    TRACY_FUNC(sceKernelGetMemBlockInfoByAddr, addr, info);
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const auto guard = std::lock_guard<std::mutex>(state->mutex);
    assert(addr >= 0);
    assert(info != nullptr);
    for (const auto &[_, block_info] : state->blocks) {
        if (block_info->mappedBase.address() <= addr && (block_info->mappedBase.address() + block_info->mappedSize > addr)) {
            memcpy(info, block_info.get(), sizeof(SceKernelMemBlockInfo));
            return SCE_KERNEL_OK;
        }
    }

    return SCE_KERNEL_ERROR_BLOCK_ERROR;
}

EXPORT(int, sceKernelGetMemBlockInfoByRange) {
    TRACY_FUNC(sceKernelGetMemBlockInfoByRange);
    return UNIMPLEMENTED();
}

EXPORT(int, sceKernelGetModel) {
    TRACY_FUNC(sceKernelGetModel);
    return emuenv.cfg.current_config.pstv_mode ? SCE_KERNEL_MODEL_VITATV : SCE_KERNEL_MODEL_VITA;
}

EXPORT(int, sceKernelGetModelForCDialog) {
    TRACY_FUNC(sceKernelGetModelForCDialog);
    return emuenv.cfg.current_config.pstv_mode ? SCE_KERNEL_MODEL_VITATV : SCE_KERNEL_MODEL_VITA;
}

EXPORT(int, sceKernelGetSubbudgetInfo) {
    TRACY_FUNC(sceKernelGetSubbudgetInfo);
    return UNIMPLEMENTED();
}

EXPORT(bool, sceKernelIsPSVitaTV) {
    TRACY_FUNC(sceKernelIsPSVitaTV);
    return emuenv.cfg.current_config.pstv_mode;
}

EXPORT(SceUID, sceKernelOpenMemBlock, const char *pName, int flags) {
    TRACY_FUNC(sceKernelOpenMemBlock, pName, flags);
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const std::lock_guard<std::mutex> memblock_lock(state->mutex);

    const auto it = std::find_if(state->blocks.begin(), state->blocks.end(), [=](const auto &block) {
        return strncmp(block.second->name, pName, KERNELOBJECT_MAX_NAME_LENGTH) == 0;
    });

    if (it != state->blocks.end())
        return it->first;

    return RET_ERROR(SCE_KERNEL_ERROR_UID_CANNOT_FIND_BY_NAME);
}

EXPORT(int, sceKernelOpenVMDomain) {
    TRACY_FUNC(sceKernelOpenVMDomain);
    return UNIMPLEMENTED();
}

EXPORT(int, sceKernelSyncVMDomain, SceUID block_uid, Address base, uint32_t size) {
    TRACY_FUNC(sceKernelSyncVMDomain, block_uid, base, size);
    const auto state = emuenv.kernel.obj_store.get<SysmemState>();
    const auto guard = std::lock_guard<std::mutex>(state->mutex);

    const auto it = state->vm_blocks.find(block_uid);
    if (it == state->vm_blocks.end()) {
        return RET_ERROR(SCE_KERNEL_ERROR_ILLEGAL_BLOCK_ID);
    }

    const auto block = it->second;
    const uint32_t block_base_end = block->mappedBase.address() + block->mappedSize;
    const uint32_t base_end = base + size;
    if (block->mappedBase.address() > base_end || base > block_base_end) {
        return RET_ERROR(SCE_KERNEL_ERROR_BLOCK_ERROR);
    }
    invalidate_jit_cache(*emuenv.kernel.get_thread(thread_id)->cpu, base, size);

    return 0;
}
