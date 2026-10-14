/**
 * @file        graphics/gl/shader.h
 * @brief       GL shader: Xenos ucode analysis + SPIR-V translations + converted GLSL.
 */
#pragma once

#include <cstdint>
#include <string>

#include <rex/graphics/pipeline/shader/spirv.h>

#include "spirv_to_glsl.h"

namespace rex::graphics::gl {

class GlShader : public SpirvShader {
 public:
  class GlTranslation : public SpirvTranslation {
   public:
    GlTranslation(GlShader& shader, uint64_t modification) : SpirvTranslation(shader, modification) {}
    // Converts the translated SPIR-V to GLSL on first use. Marks the translation invalid on failure.
    const GlslShader* GetOrCreateGlsl();
    const GlslShader* glsl() const { return glsl_converted_ ? &glsl_ : nullptr; }

   private:
    bool glsl_attempted_ = false;
    bool glsl_converted_ = false;
    GlslShader glsl_;
  };

  GlShader(xenos::ShaderType shader_type, uint64_t ucode_data_hash, const uint32_t* ucode_dwords,
           size_t ucode_dword_count, std::endian ucode_source_endian = std::endian::big)
      : SpirvShader(shader_type, ucode_data_hash, ucode_dwords, ucode_dword_count,
                    ucode_source_endian) {}

 protected:
  Translation* CreateTranslationInstance(uint64_t modification) override {
    return new GlTranslation(*this, modification);
  }
};

}  // namespace rex::graphics::gl
