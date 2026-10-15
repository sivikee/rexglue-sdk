/**
 * @file        graphics/gl/spirv_to_glsl.h
 * @brief       Xenos translator SPIR-V (Vulkan flavour) -> GLSL 4.30 through SPIRV-Cross.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rex::graphics::gl {

// GL binding point layout shared with the draw code (flattened Vulkan set * 8 + binding).
constexpr uint32_t kUboBindingSystem = 8 + 0;        // set 1, binding 0
constexpr uint32_t kUboBindingFloatVertex = 8 + 1;
constexpr uint32_t kUboBindingFloatPixel = 8 + 2;
constexpr uint32_t kUboBindingBoolLoop = 8 + 3;
constexpr uint32_t kUboBindingFetch = 8 + 4;
constexpr uint32_t kSsboBindingSharedMemory = 0;     // 4 consecutive bindings
constexpr uint32_t kUboBindingPushConstants = 15;
// Texture units: vertex shader 0.., pixel shader from kPixelTextureUnitBase.
constexpr uint32_t kPixelTextureUnitBase = 32;
// Shared memory is read through 2 R32UI texture buffers (store + page table) on these (vertex-stage-visible) units.
constexpr uint32_t kSharedMemoryTboUnitBase = 30;  // store on 30, page table on 31

struct CombinedSampler {
  uint32_t unit;             // GL texture unit (already offset for the stage)
  int32_t texture_binding;   // index into SpirvShader::GetTextureBindingsAfterTranslation()
  int32_t sampler_binding;   // index into SpirvShader::GetSamplerBindingsAfterTranslation(), -1 = dummy
};

struct GlslShader {
  std::string source;
  std::vector<CombinedSampler> samplers;
};

// texture_unit_base: first GL texture unit of this stage. texture_count: number of image bindings
// in the translated shader (samplers are located after the images in the SPIR-V binding space).
bool ConvertSpirvToGlsl(const std::vector<uint8_t>& spirv, uint32_t texture_unit_base,
                        uint32_t texture_count, GlslShader& out, std::string& error);

}  // namespace rex::graphics::gl
