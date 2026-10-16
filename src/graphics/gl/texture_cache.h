/**
 * @file        graphics/gl/texture_cache.h
 * @brief       Simplified guest texture cache for the GL backend: CPU untile/endian-swap/convert to GL
 *              formats (DXT/DXN stay block compressed), content-hash based invalidation (the Switch memory
 *              layer has no write watches), sampler objects.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glad/glad.h>

#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/register_file.h>
#include <rex/memory.h>

#include "spirv_to_glsl.h"

namespace rex::graphics::gl {

class GlRenderTargets;
class StreamRing;

class GlTextureCache {
 public:
  struct FetchInfo {
    uint8_t swizzled_signs = 0;
    uint32_t host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  };

  GlTextureCache(const RegisterFile& register_file, memory::Memory& memory)
      : regs_(register_file), memory_(memory) {}
  ~GlTextureCache() { Shutdown(); }

  bool Initialize();
  void Shutdown();
  void EndFrame();
  void SetRenderTargets(GlRenderTargets* render_targets) { render_targets_ = render_targets; }
  void SetStreamRing(StreamRing* ring) { ring_ = ring; }

  // Binds textures and samplers for one shader stage (texture units from CombinedSampler::unit).
  // fetch_info receives data for the system constants for every fetch constant used.
  void BindStage(const std::vector<CombinedSampler>& combined,
                 const std::vector<SpirvShader::TextureBinding>& texture_bindings,
                 const std::vector<SpirvShader::SamplerBinding>& sampler_bindings,
                 FetchInfo fetch_info[32]);

 private:
  struct Texture {
    GLuint id = 0;
    GLenum target = GL_TEXTURE_2D_ARRAY;
    TextureInfo info;
    uint32_t host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
    uint64_t content_hash = 0;
    uint64_t checked_frame = UINT64_MAX;
    bool valid = false;
    uint64_t last_used_frame = 0;
  };
  struct SamplerKey {
    uint32_t value[2];
    bool operator==(const SamplerKey& o) const { return value[0] == o.value[0] && value[1] == o.value[1]; }
  };
  struct SamplerKeyHash {
    size_t operator()(const SamplerKey& k) const { return k.value[0] * 0x9E3779B1u ^ k.value[1]; }
  };

  Texture* GetTexture(const xenos::xe_gpu_texture_fetch_t& fetch);
  bool Upload(Texture& texture);
  GLuint GetSampler(const xenos::xe_gpu_texture_fetch_t& fetch,
                    const SpirvShader::SamplerBinding* binding);

  const RegisterFile& regs_;
  memory::Memory& memory_;
  uint64_t frame_ = 0;
  GlRenderTargets* render_targets_ = nullptr;
  StreamRing* ring_ = nullptr;
  Texture snapshot_tex_;
  // Top-left parts of larger resolve snapshots (gl_snapshot_subrect), keyed by snapshot texture + size.
  struct SubSnapshot {
    GLuint texture = 0;
    GLenum format = 0;
    uint64_t serial = UINT64_MAX;
    uint64_t last_used_frame = 0;
  };
  std::unordered_map<uint64_t, SubSnapshot> sub_snapshots_;
  std::unordered_map<uint64_t, std::unique_ptr<Texture>> textures_;
  std::unordered_map<SamplerKey, GLuint, SamplerKeyHash> samplers_;
  GLuint dummy_2d_array_ = 0, dummy_3d_ = 0, dummy_cube_ = 0, dummy_sampler_ = 0;
  float max_anisotropy_ = 1.0f;
  uint32_t unsupported_logged_ = 0;
};

}  // namespace rex::graphics::gl
