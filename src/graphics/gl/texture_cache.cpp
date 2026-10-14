#include "texture_cache.h"

#include <algorithm>
#include <cstring>

#include <xxhash.h>

#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>

#include "gl_ext.h"

namespace rex::graphics::gl {

namespace {

enum class Convert { kCopy, k565, k1555, k4444, kDepth24, kDepth24Float };

struct HostFormat {
  GLenum internal;
  GLenum format;
  GLenum type;
  uint32_t out_bytes_per_block;
  bool compressed;
  uint32_t swizzle;
  Convert convert;
};

bool GetHostFormat(xenos::TextureFormat format, HostFormat& out) {
  using F = xenos::TextureFormat;
  const uint32_t RGBA = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  const uint32_t RRRR = xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR;
  const uint32_t RGGG = XE_GPU_MAKE_TEXTURE_SWIZZLE(R, G, G, G);
  switch (format) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_8_8_8_8_AS_16_16_16_16:
      out = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, RGBA, Convert::kCopy};
      return true;
    case F::k_8:
    case F::k_8_A:
    case F::k_8_B:
      out = {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1, false, RRRR, Convert::kCopy};
      return true;
    case F::k_8_8:
      out = {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2, false, RGGG, Convert::kCopy};
      return true;
    case F::k_5_6_5:
      out = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, RGBA, Convert::k565};
      return true;
    case F::k_1_5_5_5:
      out = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, RGBA, Convert::k1555};
      return true;
    case F::k_4_4_4_4:
      out = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, RGBA, Convert::k4444};
      return true;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_16_16_16_16:
      out = {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, 4, false, RGBA, Convert::kCopy};
      return true;
    case F::k_16:
      out = {GL_R16, GL_RED, GL_UNSIGNED_SHORT, 2, false, RRRR, Convert::kCopy};
      return true;
    case F::k_16_16:
      out = {GL_RG16, GL_RG, GL_UNSIGNED_SHORT, 4, false, RGGG, Convert::kCopy};
      return true;
    case F::k_16_16_16_16:
      out = {GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT, 8, false, RGBA, Convert::kCopy};
      return true;
    case F::k_16_FLOAT:
    case F::k_16_EXPAND:
      out = {GL_R16F, GL_RED, GL_HALF_FLOAT, 2, false, RRRR, Convert::kCopy};
      return true;
    case F::k_16_16_FLOAT:
    case F::k_16_16_EXPAND:
      out = {GL_RG16F, GL_RG, GL_HALF_FLOAT, 4, false, RGGG, Convert::kCopy};
      return true;
    case F::k_16_16_16_16_FLOAT:
    case F::k_16_16_16_16_EXPAND:
      out = {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 8, false, RGBA, Convert::kCopy};
      return true;
    case F::k_32_FLOAT:
      out = {GL_R32F, GL_RED, GL_FLOAT, 4, false, RRRR, Convert::kCopy};
      return true;
    case F::k_32_32_FLOAT:
      out = {GL_RG32F, GL_RG, GL_FLOAT, 8, false, RGGG, Convert::kCopy};
      return true;
    case F::k_32_32_32_32_FLOAT:
      out = {GL_RGBA32F, GL_RGBA, GL_FLOAT, 16, false, RGBA, Convert::kCopy};
      return true;
    case F::k_DXT1:
    case F::k_DXT1_AS_16_16_16_16:
      out = {GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 0, 0, 8, true, RGBA, Convert::kCopy};
      return true;
    case F::k_DXT2_3:
    case F::k_DXT2_3_AS_16_16_16_16:
      out = {GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, 0, 0, 16, true, RGBA, Convert::kCopy};
      return true;
    case F::k_DXT4_5:
    case F::k_DXT4_5_AS_16_16_16_16:
      out = {GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, 0, 0, 16, true, RGBA, Convert::kCopy};
      return true;
    case F::k_DXN:
      out = {GL_COMPRESSED_RG_RGTC2, 0, 0, 16, true, RGGG, Convert::kCopy};
      return true;
    case F::k_24_8:
      out = {GL_R32F, GL_RED, GL_FLOAT, 4, false, RRRR, Convert::kDepth24};
      return true;
    default:
      return false;
  }
}

void ApplyEndian(uint8_t* b, uint32_t size, xenos::Endian endian) {
  switch (endian) {
    case xenos::Endian::kNone:
      break;
    case xenos::Endian::k8in16:
      for (uint32_t o = 0; o + 2 <= size; o += 2) std::swap(b[o], b[o + 1]);
      break;
    case xenos::Endian::k8in32:
      for (uint32_t o = 0; o + 4 <= size; o += 4) {
        std::swap(b[o], b[o + 3]);
        std::swap(b[o + 1], b[o + 2]);
      }
      break;
    case xenos::Endian::k16in32:
      for (uint32_t o = 0; o + 4 <= size; o += 4) {
        std::swap(b[o], b[o + 2]);
        std::swap(b[o + 1], b[o + 3]);
      }
      break;
  }
}

uint32_t GuestToHostSwizzle(uint32_t guest_swizzle, uint32_t host_format_swizzle) {
  uint32_t host_swizzle = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t guest_component = (guest_swizzle >> (3 * i)) & 0b111;
    uint32_t host_component;
    if (guest_component >= xenos::XE_GPU_TEXTURE_SWIZZLE_0) {
      host_component = guest_component & 0b101;
    } else {
      host_component = (host_format_swizzle >> (3 * guest_component)) & 0b111;
    }
    host_swizzle |= host_component << (3 * i);
  }
  return host_swizzle;
}

inline uint8_t Expand5(uint32_t v) { return uint8_t((v << 3) | (v >> 2)); }
inline uint8_t Expand6(uint32_t v) { return uint8_t((v << 2) | (v >> 4)); }
inline uint8_t Expand4(uint32_t v) { return uint8_t((v << 4) | v); }

}  // namespace

bool GlTextureCache::Initialize() {
  uint8_t zeros[16] = {};
  glGenTextures(1, &dummy_2d_array_);
  glBindTexture(GL_TEXTURE_2D_ARRAY, dummy_2d_array_);
  glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 1, 1, 1);
  glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, zeros);
  glGenTextures(1, &dummy_3d_);
  glBindTexture(GL_TEXTURE_3D, dummy_3d_);
  glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, 1, 1, 1);
  glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, zeros);
  glGenTextures(1, &dummy_cube_);
  glBindTexture(GL_TEXTURE_CUBE_MAP, dummy_cube_);
  glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGBA8, 1, 1);
  for (int i = 0; i < 6; ++i) {
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, zeros);
  }
  glGenSamplers(1, &dummy_sampler_);
  glSamplerParameteri(dummy_sampler_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glSamplerParameteri(dummy_sampler_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_anisotropy_);
  if (glGetError() != GL_NO_ERROR) {
    max_anisotropy_ = 1.0f;
  }
  return true;
}

void GlTextureCache::Shutdown() {
  for (auto& t : textures_) {
    if (t.second->id) {
      glDeleteTextures(1, &t.second->id);
    }
  }
  textures_.clear();
  for (auto& s : samplers_) {
    glDeleteSamplers(1, &s.second);
  }
  samplers_.clear();
  GLuint* dummies[] = {&dummy_2d_array_, &dummy_3d_, &dummy_cube_};
  for (GLuint* d : dummies) {
    if (*d) {
      glDeleteTextures(1, d);
      *d = 0;
    }
  }
  if (dummy_sampler_) {
    glDeleteSamplers(1, &dummy_sampler_);
    dummy_sampler_ = 0;
  }
}

GlTextureCache::Texture* GlTextureCache::GetTexture(const xenos::xe_gpu_texture_fetch_t& fetch) {
  TextureInfo info;
  if (!TextureInfo::Prepare(fetch, &info)) {
    return nullptr;
  }
  if (!info.memory.base_address || info.memory.base_address >= 0x20000000) {
    return nullptr;
  }
  if (info.dimension != xenos::DataDimension::k2DOrStacked || info.is_stacked) {
    if (unsupported_logged_++ < 8) {
      REXGPU_WARN("GL textures: unsupported dimension {} (stacked {}) base={:08X}", uint32_t(info.dimension),
                  info.is_stacked, info.memory.base_address);
    }
    return nullptr;
  }
  uint64_t key = info.hash();
  auto it = textures_.find(key);
  Texture* tex;
  if (it == textures_.end()) {
    auto t = std::make_unique<Texture>();
    t->info = info;
    tex = t.get();
    textures_.emplace(key, std::move(t));
  } else {
    tex = it->second.get();
  }
  tex->last_used_frame = frame_;
  if (tex->checked_frame == frame_ && tex->valid) {
    return tex;
  }
  // Content check (no write watches on the Switch memory layer).
  uint64_t h = 0;
  auto hash_range = [&](uint32_t address, uint32_t size) {
    if (!address || address >= 0x20000000) {
      return;
    }
    size = std::min<uint32_t>(size, 0x20000000 - address);
    size = std::min<uint32_t>(size, 64u << 20);
    h ^= XXH3_64bits(memory_.TranslatePhysical(address), size) + 0x9E3779B97F4A7C15ull + (h << 6);
  };
  hash_range(info.memory.base_address, info.memory.base_size);
  if (info.mip_max_level > 0) {
    hash_range(info.memory.mip_address, info.memory.mip_size);
  }
  tex->checked_frame = frame_;
  if (!tex->valid || h != tex->content_hash) {
    tex->content_hash = h;
    tex->valid = Upload(*tex);
  }
  return tex->valid ? tex : nullptr;
}

bool GlTextureCache::Upload(Texture& t) {
  const TextureInfo& info = t.info;
  HostFormat hf;
  xenos::TextureFormat base_format = GetBaseFormat(info.format);
  if (!GetHostFormat(base_format, hf)) {
    if (unsupported_logged_++ < 16) {
      REXGPU_WARN("GL textures: unsupported guest format {} base={:08X}", uint32_t(info.format),
                  info.memory.base_address);
    }
    return false;
  }
  const FormatInfo* fi = info.format_info();
  uint32_t w = info.width + 1, h = info.height + 1;
  uint32_t max_levels = 1;
  for (uint32_t s = std::max(w, h); s > 1; s >>= 1) ++max_levels;
  uint32_t levels = std::min(1 + info.mip_max_level, max_levels);
  if (!t.id) {
    glGenTextures(1, &t.id);
    glBindTexture(GL_TEXTURE_2D_ARRAY, t.id);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, levels, hf.internal, w, h, 1);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, levels - 1);
    t.host_swizzle = hf.swizzle;
  } else {
    glBindTexture(GL_TEXTURE_2D_ARRAY, t.id);
  }
  uint32_t bpb = fi->bytes_per_block();
  uint32_t bpb_log2 = bpb >= 16 ? 4 : bpb >= 8 ? 3 : bpb >= 4 ? 2 : bpb >= 2 ? 1 : 0;
  std::vector<uint8_t> out;
  for (uint32_t level = 0; level < levels; ++level) {
    uint32_t mip_w = std::max(1u, w >> level), mip_h = std::max(1u, h >> level);
    uint32_t blocks_w = (mip_w + fi->block_width - 1) / fi->block_width;
    uint32_t blocks_h = (mip_h + fi->block_height - 1) / fi->block_height;
    uint32_t ox = 0, oy = 0;
    uint32_t address = info.GetMipLocation(level, &ox, &oy, true);
    if (!address || address >= 0x20000000) {
      // No data for this level: leave it zero-initialised.
      break;
    }
    TextureExtent extent = info.GetMipExtent(level, true);
    uint32_t pitch_blocks = extent.block_pitch_h;
    const uint8_t* mem = memory_.TranslatePhysical(address);
    uint32_t out_stride = blocks_w * hf.out_bytes_per_block;
    out.assign(size_t(out_stride) * blocks_h, 0);
    uint8_t block[16];
    for (uint32_t by = 0; by < blocks_h; ++by) {
      for (uint32_t bx = 0; bx < blocks_w; ++bx) {
        uint32_t off;
        if (info.is_tiled) {
          off = uint32_t(texture_util::GetTiledOffset2D(int32_t(bx + ox), int32_t(by + oy), pitch_blocks, bpb_log2));
        } else {
          off = ((by + oy) * pitch_blocks + bx + ox) * bpb;
        }
        if (address + off + bpb > 0x20000000) {
          continue;
        }
        std::memcpy(block, mem + off, bpb);
        ApplyEndian(block, bpb, info.endianness);
        uint8_t* dst = out.data() + size_t(by) * out_stride + size_t(bx) * hf.out_bytes_per_block;
        switch (hf.convert) {
          case Convert::kCopy:
            std::memcpy(dst, block, bpb);
            break;
          case Convert::k565: {
            uint32_t v = block[0] | (block[1] << 8);
            dst[0] = Expand5(v & 31);
            dst[1] = Expand6((v >> 5) & 63);
            dst[2] = Expand5(v >> 11);
            dst[3] = 255;
            break;
          }
          case Convert::k1555: {
            uint32_t v = block[0] | (block[1] << 8);
            dst[0] = Expand5(v & 31);
            dst[1] = Expand5((v >> 5) & 31);
            dst[2] = Expand5((v >> 10) & 31);
            dst[3] = (v >> 15) ? 255 : 0;
            break;
          }
          case Convert::k4444: {
            uint32_t v = block[0] | (block[1] << 8);
            dst[0] = Expand4(v & 15);
            dst[1] = Expand4((v >> 4) & 15);
            dst[2] = Expand4((v >> 8) & 15);
            dst[3] = Expand4(v >> 12);
            break;
          }
          case Convert::kDepth24:
          case Convert::kDepth24Float: {
            uint32_t v = block[0] | (block[1] << 8) | (block[2] << 16) | (uint32_t(block[3]) << 24);
            float f = xenos::UNorm24To32(v >> 8);
            std::memcpy(dst, &f, 4);
            break;
          }
        }
      }
    }
    if (hf.compressed) {
      glCompressedTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, 0, 0, 0, mip_w, mip_h, 1, hf.internal,
                                GLsizei(out.size()), out.data());
    } else {
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexSubImage3D(GL_TEXTURE_2D_ARRAY, level, 0, 0, 0, mip_w, mip_h, 1, hf.format, hf.type, out.data());
    }
  }
  t.host_swizzle = hf.swizzle;
  return true;
}

GLuint GlTextureCache::GetSampler(const xenos::xe_gpu_texture_fetch_t& fetch,
                                  const SpirvShader::SamplerBinding* binding) {
  xenos::ClampMode cx, cy, cz;
  texture_util::GetClampModesForDimension(fetch, cx, cy, cz);
  xenos::TextureFilter mag = fetch.mag_filter, min = fetch.min_filter, mip = fetch.mip_filter;
  xenos::AnisoFilter aniso = fetch.aniso_filter;
  if (binding) {
    if (binding->mag_filter != xenos::TextureFilter::kUseFetchConst) mag = binding->mag_filter;
    if (binding->min_filter != xenos::TextureFilter::kUseFetchConst) min = binding->min_filter;
    if (binding->mip_filter != xenos::TextureFilter::kUseFetchConst) mip = binding->mip_filter;
    if (binding->aniso_filter != xenos::AnisoFilter::kUseFetchConst) aniso = binding->aniso_filter;
  }
  uint32_t aniso_value = uint32_t(aniso) > 5 ? 0 : uint32_t(aniso);
  SamplerKey key;
  key.value[0] = uint32_t(mag) | (uint32_t(min) << 2) | (uint32_t(mip) << 4) | (aniso_value << 6) |
                 (uint32_t(cx) << 9) | (uint32_t(cy) << 12) | (uint32_t(cz) << 15) |
                 (uint32_t(fetch.border_color) << 18);
  key.value[1] = 0;
  auto it = samplers_.find(key);
  if (it != samplers_.end()) {
    return it->second;
  }
  GLuint s;
  glGenSamplers(1, &s);
  bool mag_linear = mag == xenos::TextureFilter::kLinear;
  bool min_linear = min == xenos::TextureFilter::kLinear;
  GLenum min_filter;
  if (mip == xenos::TextureFilter::kBaseMap) {
    min_filter = min_linear ? GL_LINEAR : GL_NEAREST;
  } else if (mip == xenos::TextureFilter::kLinear) {
    min_filter = min_linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_LINEAR;
  } else {
    min_filter = min_linear ? GL_LINEAR_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_NEAREST;
  }
  glSamplerParameteri(s, GL_TEXTURE_MIN_FILTER, min_filter);
  glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, mag_linear ? GL_LINEAR : GL_NEAREST);
  auto wrap = [](xenos::ClampMode m) -> GLenum {
    switch (m) {
      case xenos::ClampMode::kRepeat: return GL_REPEAT;
      case xenos::ClampMode::kMirroredRepeat: return GL_MIRRORED_REPEAT;
      case xenos::ClampMode::kClampToBorder:
      case xenos::ClampMode::kMirrorClampToBorder: return GL_CLAMP_TO_BORDER;
      case xenos::ClampMode::kMirrorClampToEdge:
      case xenos::ClampMode::kMirrorClampToHalfway: return GL_MIRROR_CLAMP_TO_EDGE;
      default: return GL_CLAMP_TO_EDGE;
    }
  };
  glSamplerParameteri(s, GL_TEXTURE_WRAP_S, wrap(cx));
  glSamplerParameteri(s, GL_TEXTURE_WRAP_T, wrap(cy));
  glSamplerParameteri(s, GL_TEXTURE_WRAP_R, wrap(cz));
  float border[4] = {0, 0, 0, 0};
  if (fetch.border_color == xenos::BorderColor::k_ABGR_White) {
    border[0] = border[1] = border[2] = border[3] = 1.0f;
  }
  glSamplerParameterfv(s, GL_TEXTURE_BORDER_COLOR, border);
  if (aniso_value >= 1 && max_anisotropy_ > 1.0f) {
    float a = std::min(float(1u << (aniso_value - 1)), max_anisotropy_);
    glSamplerParameterf(s, GL_TEXTURE_MAX_ANISOTROPY_EXT, a);
  }
  samplers_.emplace(key, s);
  return s;
}

void GlTextureCache::BindStage(const std::vector<CombinedSampler>& combined,
                               const std::vector<SpirvShader::TextureBinding>& texture_bindings,
                               const std::vector<SpirvShader::SamplerBinding>& sampler_bindings,
                               FetchInfo fetch_info[32]) {
  for (const CombinedSampler& cs : combined) {
    if (cs.texture_binding < 0 || size_t(cs.texture_binding) >= texture_bindings.size()) {
      continue;
    }
    const SpirvShader::TextureBinding& tb = texture_bindings[size_t(cs.texture_binding)];
    xenos::xe_gpu_texture_fetch_t fetch = regs_.GetTextureFetch(tb.fetch_constant);
    GLenum target;
    GLuint dummy;
    switch (tb.dimension) {
      case xenos::FetchOpDimension::k3DOrStacked:
        target = GL_TEXTURE_3D;
        dummy = dummy_3d_;
        break;
      case xenos::FetchOpDimension::kCube:
        target = GL_TEXTURE_CUBE_MAP;
        dummy = dummy_cube_;
        break;
      default:
        target = GL_TEXTURE_2D_ARRAY;
        dummy = dummy_2d_array_;
        break;
    }
    Texture* tex = nullptr;
    if (fetch.type == xenos::FetchConstantType::kTexture) {
      tex = GetTexture(fetch);
      if (tex && tex->target != target) {
        tex = nullptr;
      }
    }
    FetchInfo& fi = fetch_info[tb.fetch_constant];
    fi.swizzled_signs = texture_util::SwizzleSigns(fetch);
    fi.host_swizzle = tex ? GuestToHostSwizzle(fetch.swizzle, tex->host_swizzle)
                          : GuestToHostSwizzle(fetch.swizzle, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA);
    glActiveTexture(GL_TEXTURE0 + cs.unit);
    glBindTexture(target, tex ? tex->id : dummy);
    GLuint sampler = dummy_sampler_;
    if (tex) {
      const SpirvShader::SamplerBinding* sb = nullptr;
      if (cs.sampler_binding >= 0 && size_t(cs.sampler_binding) < sampler_bindings.size()) {
        sb = &sampler_bindings[size_t(cs.sampler_binding)];
      }
      sampler = GetSampler(fetch, sb);
    }
    glBindSampler(cs.unit, sampler);
  }
}

}  // namespace rex::graphics::gl
