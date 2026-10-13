/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/graphics/vulkan/shared_memory.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_BOOL(vulkan_sparse_shared_memory, true, "GPU/Vulkan",
                    "Use sparse shared memory on Vulkan")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);
REXCVAR_DEFINE_BOOL(vulkan_hoist_shared_memory_uploads, true, "GPU/Vulkan",
                    "Upload guest memory not yet accessed in the current submission at the "
                    "beginning of the submission instead of between draws - avoids splitting "
                    "render passes, which is expensive on tile-based GPUs");
REXCVAR_DEFINE_BOOL(vulkan_import_guest_memory, false, "GPU/Vulkan",
                    "Alias the guest physical memory as the shared memory buffer when "
                    "VK_EXT_external_memory_host is available (unified memory) instead of "
                    "uploading - avoids splitting render passes for every upload")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

namespace rex::graphics::vulkan {

VulkanSharedMemory::VulkanSharedMemory(VulkanCommandProcessor& command_processor,
                                       memory::Memory& memory,
                                       VkPipelineStageFlags guest_shader_pipeline_stages)
    : SharedMemory(memory),
      command_processor_(command_processor),
      guest_shader_pipeline_stages_(guest_shader_pipeline_stages) {}

VulkanSharedMemory::~VulkanSharedMemory() {
  Shutdown(true);
}

bool VulkanSharedMemory::Initialize() {
  InitializeCommon();

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  const VkBufferCreateFlags sparse_flags =
      VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;

  // Try to create a sparse buffer.
  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = sparse_flags;
  buffer_create_info.size = kBufferSize;
  buffer_create_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = nullptr;
  if (REXCVAR_GET(vulkan_import_guest_memory) &&
      vulkan_device->extensions().ext_EXT_external_memory_host) {
    if (ImportGuestMemory(buffer_create_info)) {
      REXGPU_INFO("Shared memory: aliasing the guest physical memory (no uploads)");
    } else {
      REXGPU_WARN("Shared memory: failed to import the guest memory, uploading instead");
    }
  }
  if (buffer_ == VK_NULL_HANDLE && REXCVAR_GET(vulkan_sparse_shared_memory) &&
      vulkan_device->properties().sparseResidencyBuffer) {
    if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_) == VK_SUCCESS) {
      VkMemoryRequirements buffer_memory_requirements;
      dfn.vkGetBufferMemoryRequirements(device, buffer_, &buffer_memory_requirements);
      if (rex::bit_scan_forward(buffer_memory_requirements.memoryTypeBits &
                                    vulkan_device->memory_types().device_local,
                                &buffer_memory_type_)) {
        uint32_t allocation_size_log2;
        rex::bit_scan_forward(std::max(uint64_t(buffer_memory_requirements.alignment), uint64_t(1)),
                              &allocation_size_log2);
        if (allocation_size_log2 < kBufferSizeLog2) {
          // Maximum of 1024 allocations in the worst case for all of the
          // buffer because of the overall 4096 allocation count limit on
          // Windows drivers.
          InitializeSparseHostGpuMemory(std::max(
              allocation_size_log2,
              std::max(kHostGpuMemoryOptimalSparseAllocationLog2, kBufferSizeLog2 - uint32_t(10))));
        } else {
          // Shouldn't happen on any real platform, but no point allocating the
          // buffer sparsely.
          dfn.vkDestroyBuffer(device, buffer_, nullptr);
          buffer_ = VK_NULL_HANDLE;
        }
      } else {
        REXGPU_ERROR(
            "Shared memory: Failed to get a device-local Vulkan memory type "
            "for the sparse buffer");
        dfn.vkDestroyBuffer(device, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
      }
    } else {
      REXGPU_ERROR("Shared memory: Failed to create the {} MB Vulkan sparse buffer",
                   kBufferSize >> 20);
    }
  }

  // Create a non-sparse buffer if there were issues with the sparse buffer.
  if (buffer_ == VK_NULL_HANDLE) {
    REXGPU_INFO(
        "Vulkan sparse binding is not used for shared memory emulation - video "
        "memory usage may increase significantly because a full {} MB buffer "
        "will be created",
        kBufferSize >> 20);
    buffer_create_info.flags &= ~sparse_flags;
    if (dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer_) != VK_SUCCESS) {
      REXGPU_ERROR("Shared memory: Failed to create the {} MB Vulkan buffer", kBufferSize >> 20);
      Shutdown();
      return false;
    }
    VkMemoryRequirements buffer_memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, buffer_, &buffer_memory_requirements);
    if (!rex::bit_scan_forward(
            buffer_memory_requirements.memoryTypeBits & vulkan_device->memory_types().device_local,
            &buffer_memory_type_)) {
      REXGPU_ERROR(
          "Shared memory: Failed to get a device-local Vulkan memory type for "
          "the buffer");
      Shutdown();
      return false;
    }
    VkMemoryAllocateInfo buffer_memory_allocate_info;
    VkMemoryAllocateInfo* buffer_memory_allocate_info_last = &buffer_memory_allocate_info;
    buffer_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    buffer_memory_allocate_info.pNext = nullptr;
    buffer_memory_allocate_info.allocationSize = buffer_memory_requirements.size;
    buffer_memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
    VkMemoryDedicatedAllocateInfo buffer_memory_dedicated_allocate_info;
    if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
      buffer_memory_allocate_info_last->pNext = &buffer_memory_dedicated_allocate_info;
      buffer_memory_allocate_info_last =
          reinterpret_cast<VkMemoryAllocateInfo*>(&buffer_memory_dedicated_allocate_info);
      buffer_memory_dedicated_allocate_info.sType =
          VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
      buffer_memory_dedicated_allocate_info.pNext = nullptr;
      buffer_memory_dedicated_allocate_info.image = VK_NULL_HANDLE;
      buffer_memory_dedicated_allocate_info.buffer = buffer_;
    }
    VkDeviceMemory buffer_memory;
    if (dfn.vkAllocateMemory(device, &buffer_memory_allocate_info, nullptr, &buffer_memory) !=
        VK_SUCCESS) {
      REXGPU_ERROR(
          "Shared memory: Failed to allocate {} MB of memory for the Vulkan "
          "buffer",
          kBufferSize >> 20);
      Shutdown();
      return false;
    }
    buffer_memory_.push_back(buffer_memory);
    if (dfn.vkBindBufferMemory(device, buffer_, buffer_memory, 0) != VK_SUCCESS) {
      REXGPU_ERROR("Shared memory: Failed to bind memory to the Vulkan buffer");
      Shutdown();
      return false;
    }
  }

  // The first usage will likely be uploading.
  last_usage_ = Usage::kTransferDestination;
  last_written_range_ = std::make_pair<uint32_t, uint32_t>(0, 0);

  upload_buffer_pool_ = std::make_unique<ui::vulkan::VulkanUploadBufferPool>(
      vulkan_device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      rex::align(ui::vulkan::VulkanUploadBufferPool::kDefaultPageSize, size_t(1)
                                                                           << page_size_log2()));

  return true;
}

void VulkanSharedMemory::Shutdown(bool from_destructor) {
  upload_buffer_pool_.reset();

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, buffer_);
  for (VkDeviceMemory memory : buffer_memory_) {
    dfn.vkFreeMemory(device, memory, nullptr);
  }
  buffer_memory_.clear();
  guest_memory_imported_ = false;

  // If calling from the destructor, the SharedMemory destructor will call
  // ShutdownCommon.
  if (!from_destructor) {
    ShutdownCommon();
  }
}

void VulkanSharedMemory::CompletedSubmissionUpdated() {
  upload_buffer_pool_->Reclaim(command_processor_.GetCompletedSubmission());
}

void VulkanSharedMemory::EndSubmission() {
  upload_buffer_pool_->FlushWrites();
}

void VulkanSharedMemory::Use(Usage usage, std::pair<uint32_t, uint32_t> written_range) {
  written_range.first = std::min(written_range.first, kBufferSize);
  written_range.second = std::min(written_range.second, kBufferSize - written_range.first);
  assert_true(usage != Usage::kRead || !written_range.second);
  if (last_usage_ != usage || last_written_range_.second) {
    VkPipelineStageFlags src_stage_mask, dst_stage_mask;
    VkAccessFlags src_access_mask, dst_access_mask;
    GetUsageMasks(last_usage_, src_stage_mask, src_access_mask);
    GetUsageMasks(usage, dst_stage_mask, dst_access_mask);
    VkDeviceSize offset, size;
    if (last_usage_ == usage) {
      // Committing the previous write, while not changing the access mask
      // (passing false as whether to skip the barrier if no masks are changed
      // for this reason).
      offset = VkDeviceSize(last_written_range_.first);
      size = VkDeviceSize(last_written_range_.second);
    } else {
      // Changing the stage and access mask - all preceding writes must be
      // available not only to the source stage, but to the destination as well.
      offset = 0;
      size = VK_WHOLE_SIZE;
      last_usage_ = usage;
    }
    command_processor_.PushBufferMemoryBarrier(
        buffer_, offset, size, src_stage_mask, dst_stage_mask, src_access_mask, dst_access_mask,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  }
  last_written_range_ = written_range;
}

bool VulkanSharedMemory::AllocateSparseHostGpuMemoryRange(uint32_t offset_allocations,
                                                          uint32_t length_allocations) {
  if (!length_allocations) {
    return true;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  VkMemoryAllocateInfo memory_allocate_info;
  memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memory_allocate_info.pNext = nullptr;
  memory_allocate_info.allocationSize = length_allocations
                                        << host_gpu_memory_sparse_granularity_log2();
  memory_allocate_info.memoryTypeIndex = buffer_memory_type_;
  VkDeviceMemory memory;
  if (dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &memory) != VK_SUCCESS) {
    REXGPU_ERROR("Shared memory: Failed to allocate sparse buffer memory");
    return false;
  }
  buffer_memory_.push_back(memory);

  VkSparseMemoryBind bind;
  bind.resourceOffset = offset_allocations << host_gpu_memory_sparse_granularity_log2();
  bind.size = memory_allocate_info.allocationSize;
  bind.memory = memory;
  bind.memoryOffset = 0;
  bind.flags = 0;
  VkPipelineStageFlags bind_wait_stage_mask =
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (vulkan_device->properties().tessellationShader) {
    bind_wait_stage_mask |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
  }
  command_processor_.SparseBindBuffer(buffer_, 1, &bind, bind_wait_stage_mask);

  return true;
}

bool VulkanSharedMemory::UploadRanges(
    const std::vector<std::pair<uint32_t, uint32_t>>& upload_page_ranges) {
  if (upload_page_ranges.empty()) {
    return true;
  }
  if (guest_memory_imported_) {
    // The buffer is the guest memory - nothing to copy, and host writes are
    // made visible to the device by the queue submission.
    for (auto upload_range : upload_page_ranges) {
      MakeRangeValid(upload_range.first << page_size_log2(),
                     upload_range.second << page_size_log2(), false);
    }
    return true;
  }

  // Uploads of pages not accessed by the GPU yet in this submission are done at
  // the beginning of the submission instead of between draws - copying ends the
  // render pass, which is very expensive on tile-based GPUs (the whole render
  // target is stored and loaded again). The data is still snapshotted now.
  bool hoist = REXCVAR_GET(vulkan_hoist_shared_memory_uploads);
  stream_upload_page_ranges_.clear();
  bool successful = true;
  for (auto upload_range : upload_page_ranges) {
    if (hoist && !AreSubmissionPagesTouched(upload_range.first, upload_range.second)) {
      successful &= StageUploadRange(upload_range.first, upload_range.second, true);
    } else {
      stream_upload_page_ranges_.push_back(upload_range);
    }
  }

  if (!stream_upload_page_ranges_.empty()) {
    // upload_page_ranges are sorted, use them to determine the range for the
    // ordering barrier.
    Use(Usage::kTransferDestination,
        std::make_pair(stream_upload_page_ranges_.front().first << page_size_log2(),
                       (stream_upload_page_ranges_.back().first +
                        stream_upload_page_ranges_.back().second -
                        stream_upload_page_ranges_.front().first)
                           << page_size_log2()));
    command_processor_.SubmitBarriers(true);
    upload_regions_.clear();
    upload_buffer_previous_ = VK_NULL_HANDLE;
    for (auto upload_range : stream_upload_page_ranges_) {
      if (!StageUploadRange(upload_range.first, upload_range.second, false)) {
        successful = false;
        break;
      }
    }
    FlushStreamUploadRegions();
  }

  for (auto upload_range : upload_page_ranges) {
    TouchSubmissionPages(upload_range.first, upload_range.second);
  }
  return successful;
}

bool VulkanSharedMemory::StageUploadRange(uint32_t page_start, uint32_t page_count,
                                          bool hoisted) {
  uint64_t submission_current = command_processor_.GetCurrentSubmission();
  while (page_count) {
    VkBuffer upload_buffer;
    VkDeviceSize upload_buffer_offset, upload_buffer_size;
    uint8_t* upload_buffer_mapping = upload_buffer_pool_->RequestPartial(
        submission_current, page_count << page_size_log2(), size_t(1) << page_size_log2(),
        upload_buffer, upload_buffer_offset, upload_buffer_size);
    if (upload_buffer_mapping == nullptr) {
      REXGPU_ERROR("Shared memory: Failed to get a Vulkan upload buffer");
      return false;
    }
    MakeRangeValid(page_start << page_size_log2(), uint32_t(upload_buffer_size), false);
    std::memcpy(upload_buffer_mapping, memory().TranslatePhysical(page_start << page_size_log2()),
                upload_buffer_size);
    VkBufferCopy upload_region;
    upload_region.srcOffset = upload_buffer_offset;
    upload_region.dstOffset = VkDeviceSize(page_start << page_size_log2());
    upload_region.size = upload_buffer_size;
    if (hoisted) {
      hoisted_uploads_.push_back({upload_buffer, upload_region});
    } else {
      if (upload_buffer_previous_ != upload_buffer) {
        FlushStreamUploadRegions();
      }
      upload_buffer_previous_ = upload_buffer;
      upload_regions_.push_back(upload_region);
    }
    uint32_t upload_buffer_pages = uint32_t(upload_buffer_size >> page_size_log2());
    page_start += upload_buffer_pages;
    page_count -= upload_buffer_pages;
  }
  return true;
}

void VulkanSharedMemory::FlushStreamUploadRegions() {
  if (upload_regions_.empty()) {
    return;
  }
  assert_true(upload_buffer_previous_ != VK_NULL_HANDLE);
  command_processor_.deferred_command_buffer().CmdVkCopyBuffer(
      upload_buffer_previous_, buffer_, uint32_t(upload_regions_.size()), upload_regions_.data());
  upload_regions_.clear();
}

void VulkanSharedMemory::TouchSubmissionPages(uint32_t page_start, uint32_t page_count) {
  if (submission_touched_pages_.empty()) {
    submission_touched_pages_.resize(((kBufferSize >> page_size_log2()) + 63) / 64);
  }
  for (uint32_t page = page_start; page < page_start + page_count;) {
    uint32_t bit = page & 63;
    uint32_t bits = std::min(64 - bit, page_start + page_count - page);
    uint64_t mask = bits == 64 ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1) << bit;
    submission_touched_pages_[page >> 6] |= mask;
    page += bits;
  }
}

bool VulkanSharedMemory::AreSubmissionPagesTouched(uint32_t page_start, uint32_t page_count) const {
  if (submission_touched_pages_.empty()) {
    return false;
  }
  for (uint32_t page = page_start; page < page_start + page_count;) {
    uint32_t bit = page & 63;
    uint32_t bits = std::min(64 - bit, page_start + page_count - page);
    uint64_t mask = bits == 64 ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1) << bit;
    if (submission_touched_pages_[page >> 6] & mask) {
      return true;
    }
    page += bits;
  }
  return false;
}

void VulkanSharedMemory::TouchSubmissionBytes(uint32_t start, uint32_t length) {
  if (!length || start >= kBufferSize) {
    return;
  }
  length = std::min(length, kBufferSize - start);
  uint32_t page_first = start >> page_size_log2();
  uint32_t page_last = (start + length - 1) >> page_size_log2();
  TouchSubmissionPages(page_first, page_last - page_first + 1);
}

void VulkanSharedMemory::OnRangesRequested(
    const std::vector<std::pair<uint32_t, uint32_t>>& ranges) {
  for (const std::pair<uint32_t, uint32_t>& range : ranges) {
    TouchSubmissionBytes(range.first, range.second);
  }
}

void VulkanSharedMemory::OnRangeWrittenByGpu(uint32_t start, uint32_t length) {
  TouchSubmissionBytes(start, length);
}

void VulkanSharedMemory::RecordHoistedUploads(VkCommandBuffer command_buffer) {
  if (!submission_touched_pages_.empty()) {
    std::fill(submission_touched_pages_.begin(), submission_touched_pages_.end(), 0);
  }
  if (hoisted_uploads_.empty()) {
    return;
  }
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();

  // Earlier submissions on the queue may still be reading the buffer.
  VkBufferMemoryBarrier buffer_barrier = {};
  buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  buffer_barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  buffer_barrier.buffer = buffer_;
  buffer_barrier.offset = 0;
  buffer_barrier.size = VK_WHOLE_SIZE;
  dfn.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &buffer_barrier, 0,
                           nullptr);

  std::sort(hoisted_uploads_.begin(), hoisted_uploads_.end(),
            [](const HoistedUpload& a, const HoistedUpload& b) { return a.source < b.source; });
  hoisted_upload_regions_.clear();
  for (size_t i = 0; i < hoisted_uploads_.size(); ++i) {
    hoisted_upload_regions_.push_back(hoisted_uploads_[i].region);
    if (i + 1 == hoisted_uploads_.size() ||
        hoisted_uploads_[i + 1].source != hoisted_uploads_[i].source) {
      dfn.vkCmdCopyBuffer(command_buffer, hoisted_uploads_[i].source, buffer_,
                          uint32_t(hoisted_upload_regions_.size()),
                          hoisted_upload_regions_.data());
      hoisted_upload_regions_.clear();
    }
  }
  hoisted_uploads_.clear();

  // Everything in the submission may read the uploaded data.
  buffer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  buffer_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  dfn.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &buffer_barrier,
                           0, nullptr);
}

bool VulkanSharedMemory::ImportGuestMemory(const VkBufferCreateInfo& buffer_create_info) {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  constexpr VkExternalMemoryHandleTypeFlagBits kHandleType =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

  void* host_pointer = memory().TranslatePhysical(0);
  VkMemoryHostPointerPropertiesEXT host_pointer_properties = {};
  host_pointer_properties.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
  if (dfn.vkGetMemoryHostPointerPropertiesEXT(device, kHandleType, host_pointer,
                                              &host_pointer_properties) != VK_SUCCESS) {
    return false;
  }

  VkExternalMemoryBufferCreateInfo external_memory_buffer_create_info = {};
  external_memory_buffer_create_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  external_memory_buffer_create_info.handleTypes = kHandleType;
  VkBufferCreateInfo import_buffer_create_info = buffer_create_info;
  import_buffer_create_info.pNext = &external_memory_buffer_create_info;
  import_buffer_create_info.flags = 0;
  VkBuffer buffer;
  if (dfn.vkCreateBuffer(device, &import_buffer_create_info, nullptr, &buffer) != VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements buffer_memory_requirements;
  dfn.vkGetBufferMemoryRequirements(device, buffer, &buffer_memory_requirements);
  uint32_t memory_type;
  if (buffer_memory_requirements.size > kBufferSize ||
      !rex::bit_scan_forward(
          buffer_memory_requirements.memoryTypeBits & host_pointer_properties.memoryTypeBits,
          &memory_type)) {
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    return false;
  }

  VkImportMemoryHostPointerInfoEXT import_memory_host_pointer_info = {};
  import_memory_host_pointer_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
  import_memory_host_pointer_info.handleType = kHandleType;
  import_memory_host_pointer_info.pHostPointer = host_pointer;
  VkMemoryAllocateInfo memory_allocate_info = {};
  memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memory_allocate_info.pNext = &import_memory_host_pointer_info;
  memory_allocate_info.allocationSize = kBufferSize;
  memory_allocate_info.memoryTypeIndex = memory_type;
  VkDeviceMemory buffer_memory;
  if (dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &buffer_memory) !=
      VK_SUCCESS) {
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    return false;
  }
  if (dfn.vkBindBufferMemory(device, buffer, buffer_memory, 0) != VK_SUCCESS) {
    dfn.vkFreeMemory(device, buffer_memory, nullptr);
    dfn.vkDestroyBuffer(device, buffer, nullptr);
    return false;
  }
  buffer_ = buffer;
  buffer_memory_type_ = memory_type;
  buffer_memory_.push_back(buffer_memory);
  guest_memory_imported_ = true;
  return true;
}

void VulkanSharedMemory::GetUsageMasks(Usage usage, VkPipelineStageFlags& stage_mask,
                                       VkAccessFlags& access_mask) const {
  switch (usage) {
    case Usage::kComputeWrite:
      stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      access_mask = VK_ACCESS_SHADER_READ_BIT;
      return;
    case Usage::kTransferDestination:
      stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
      return;
    default:
      break;
  }
  stage_mask = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | guest_shader_pipeline_stages_;
  access_mask = VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
  switch (usage) {
    case Usage::kRead:
      stage_mask |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask |= VK_ACCESS_TRANSFER_READ_BIT;
      break;
    case Usage::kGuestDrawReadWrite:
      access_mask |= VK_ACCESS_SHADER_WRITE_BIT;
      break;
    default:
      assert_unhandled_case(usage);
  }
}

}  // namespace rex::graphics::vulkan
