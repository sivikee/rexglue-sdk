#include "shader.h"

#include <rex/logging.h>

namespace rex::graphics::gl {

const GlslShader* GlShader::GlTranslation::GetOrCreateGlsl() {
  if (glsl_converted_) {
    return &glsl_;
  }
  if (glsl_attempted_ || !is_valid()) {
    return nullptr;
  }
  glsl_attempted_ = true;
  const GlShader& shader = static_cast<const GlShader&>(this->shader());
  uint32_t unit_base = shader.type() == xenos::ShaderType::kPixel ? kPixelTextureUnitBase : 0;
  std::string error;
  if (!ConvertSpirvToGlsl(translated_binary(), unit_base,
                          uint32_t(shader.GetTextureBindingsAfterTranslation().size()), glsl_,
                          error)) {
    REXGPU_ERROR("GL: SPIRV-Cross failed for shader {:016X} modification {:016X}: {}",
                 shader.ucode_data_hash(), modification(), error);
    MakeInvalid();
    return nullptr;
  }
  glsl_converted_ = true;
  return &glsl_;
}

}  // namespace rex::graphics::gl
