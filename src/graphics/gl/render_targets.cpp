#include "render_targets.h"
#include "perf.h"

#include <algorithm>
#include <cstring>

#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/math.h>

#include "gl_ext.h"

namespace rex::graphics::gl {

namespace {

constexpr uint32_t kMaxRtDimension = 4096;

float HalfToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign << 31;
    } else {
      exp = 1;
      while (!(man & 0x400)) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FF;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7F800000 | (man << 13);
  } else {
    f = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float r;
  std::memcpy(&r, &f, 4);
  return r;
}

uint16_t FloatToHalf(float v) {
  uint32_t f;
  std::memcpy(&f, &v, 4);
  uint32_t sign = (f >> 16) & 0x8000;
  int32_t exp = int32_t((f >> 23) & 0xFF) - 127 + 15;
  uint32_t man = f & 0x7FFFFF;
  if (((f >> 23) & 0xFF) == 0xFF) {
    return uint16_t(sign | 0x7C00 | (man ? 0x200 : 0));
  }
  if (exp >= 31) {
    return uint16_t(sign | 0x7C00);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return uint16_t(sign);
    }
    man |= 0x800000;
    uint32_t shift = uint32_t(14 - exp);
    return uint16_t(sign | (man >> shift));
  }
  return uint16_t(sign | (uint32_t(exp) << 10) | (man >> 13));
}

// Applies the Xenos endian swap to up to 16 bytes (self-inverse permutations).
void ApplyEndian(uint8_t* b, uint32_t size, xenos::Endian128 endian) {
  auto swap_units = [&](uint32_t unit) {
    for (uint32_t o = 0; o + unit <= size; o += unit) {
      std::reverse(b + o, b + o + unit);
    }
  };
  switch (endian) {
    case xenos::Endian128::kNone:
      break;
    case xenos::Endian128::k8in16:
      swap_units(2);
      break;
    case xenos::Endian128::k8in32:
      swap_units(4);
      break;
    case xenos::Endian128::k16in32:
      for (uint32_t o = 0; o + 4 <= size; o += 4) {
        std::swap(b[o], b[o + 2]);
        std::swap(b[o + 1], b[o + 3]);
      }
      break;
    case xenos::Endian128::k8in64:
      swap_units(8);
      break;
    case xenos::Endian128::k8in128:
      swap_units(16);
      break;
  }
}

}  // namespace

bool GlRenderTargets::Initialize() {
  glGenFramebuffers(1, &read_fbo_);
  return true;
}

void GlRenderTargets::Shutdown() {
  for (auto& f : framebuffers_) {
    glDeleteFramebuffers(1, &f.second);
  }
  framebuffers_.clear();
  for (auto& c : color_targets_) {
    glDeleteTextures(1, &c.second.texture);
  }
  color_targets_.clear();
  for (auto& d : depth_targets_) {
    glDeleteTextures(1, &d.second.texture);
  }
  depth_targets_.clear();
  if (read_fbo_) {
    glDeleteFramebuffers(1, &read_fbo_);
    read_fbo_ = 0;
  }
}

GLenum GlRenderTargets::GetInternalFormat(xenos::ColorRenderTargetFormat format) {
  using F = xenos::ColorRenderTargetFormat;
  switch (format) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_GAMMA:
      return GL_RGBA8;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_10_10_10_10:
      return GL_RGB10_A2;
    case F::k_32_FLOAT:
      return GL_R32F;
    case F::k_32_32_FLOAT:
      return GL_RG32F;
    default:
      return GL_RGBA16F;
  }
}

void GlRenderTargets::InvalidateFramebuffersUsing(GLuint texture) {
  for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
    const FbKey& k = it->first;
    if (k.color[0] == texture || k.color[1] == texture || k.color[2] == texture ||
        k.color[3] == texture || k.depth == texture) {
      glDeleteFramebuffers(1, &it->second);
      it = framebuffers_.erase(it);
    } else {
      ++it;
    }
  }
}

GlRenderTargets::ColorTarget* GlRenderTargets::GetColorTarget(uint32_t base,
                                                              xenos::ColorRenderTargetFormat format,
                                                              uint32_t width, uint32_t height,
                                                              bool create) {
  width = std::min(std::max(width, 1u), kMaxRtDimension);
  height = std::min(std::max(height, 1u), kMaxRtDimension);
  auto key = std::make_pair(base, uint32_t(format));
  auto it = color_targets_.find(key);
  if (it == color_targets_.end()) {
    if (!create) {
      return nullptr;
    }
    ColorTarget& t = color_targets_[key];
    t.base = base;
    t.format = format;
    t.internal_format = GetInternalFormat(format);
    t.width = width;
    t.height = height;
    glGenTextures(1, &t.texture);
    glBindTexture(GL_TEXTURE_2D, t.texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, t.internal_format, width, height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    REXGPU_INFO("GL RT: created color target base={} fmt={} {}x{}", base, uint32_t(format), width, height);
    return &t;
  }
  ColorTarget& t = it->second;
  if (create && (width > t.width || height > t.height)) {
    uint32_t new_w = std::max(width, t.width), new_h = std::max(height, t.height);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, t.internal_format, new_w, new_h);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glCopyImageSubData(t.texture, GL_TEXTURE_2D, 0, 0, 0, 0, tex, GL_TEXTURE_2D, 0, 0, 0, 0, t.width,
                       t.height, 1);
    InvalidateFramebuffersUsing(t.texture);
    glDeleteTextures(1, &t.texture);
    REXGPU_INFO("GL RT: grew color target base={} {}x{} -> {}x{}", base, t.width, t.height, new_w, new_h);
    t.texture = tex;
    t.width = new_w;
    t.height = new_h;
  }
  return &t;
}

GlRenderTargets::DepthTarget* GlRenderTargets::GetDepthTarget(uint32_t base, uint32_t width,
                                                              uint32_t height, bool create) {
  width = std::min(std::max(width, 1u), kMaxRtDimension);
  height = std::min(std::max(height, 1u), kMaxRtDimension);
  auto it = depth_targets_.find(base);
  if (it == depth_targets_.end()) {
    if (!create) {
      return nullptr;
    }
    DepthTarget& t = depth_targets_[base];
    t.base = base;
    t.width = width;
    t.height = height;
    glGenTextures(1, &t.texture);
    glBindTexture(GL_TEXTURE_2D, t.texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, width, height);
    REXGPU_INFO("GL RT: created depth target base={} {}x{}", base, width, height);
    return &t;
  }
  DepthTarget& t = it->second;
  if (create && (width > t.width || height > t.height)) {
    uint32_t new_w = std::max(width, t.width), new_h = std::max(height, t.height);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, new_w, new_h);
    glCopyImageSubData(t.texture, GL_TEXTURE_2D, 0, 0, 0, 0, tex, GL_TEXTURE_2D, 0, 0, 0, 0, t.width,
                       t.height, 1);
    InvalidateFramebuffersUsing(t.texture);
    glDeleteTextures(1, &t.texture);
    t.texture = tex;
    t.width = new_w;
    t.height = new_h;
  }
  return &t;
}

GLuint GlRenderTargets::GetFramebuffer(const FbKey& key) {
  auto it = framebuffers_.find(key);
  if (it != framebuffers_.end()) {
    return it->second;
  }
  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  GLenum draw_buffers[4];
  for (uint32_t i = 0; i < 4; ++i) {
    if (key.color[i]) {
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, key.color[i], 0);
      draw_buffers[i] = GL_COLOR_ATTACHMENT0 + i;
    } else {
      draw_buffers[i] = GL_NONE;
    }
  }
  if (key.depth) {
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, key.depth, 0);
  }
  glDrawBuffers(4, draw_buffers);
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    REXGPU_ERROR("GL RT: framebuffer incomplete: {:x}", status);
  }
  framebuffers_[key] = fbo;
  return fbo;
}

bool GlRenderTargets::BindForDraw(uint32_t color_mask, bool depth_stencil_used, uint32_t width_needed,
                                  uint32_t height_needed) {
  auto surface_info = regs_.Get<reg::RB_SURFACE_INFO>();
  uint32_t width = std::max<uint32_t>(surface_info.surface_pitch, width_needed);
  uint32_t height = std::max<uint32_t>(height_needed, 1);
  FbKey key = {};
  uint32_t used_colors = 0;
  uint32_t min_w = UINT32_MAX, min_h = UINT32_MAX;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    if (!((color_mask >> (i * 4)) & 0b1111)) {
      continue;
    }
    auto color_info = regs_.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    ColorTarget* t = GetColorTarget(color_info.color_base, color_info.color_format, width, height, true);
    if (!t) {
      continue;
    }
    t->last_used = frame_;
    key.color[i] = t->texture;
    used_colors |= 1u << i;
    min_w = std::min(min_w, t->width);
    min_h = std::min(min_h, t->height);
  }
  if (depth_stencil_used) {
    auto depth_info = regs_.Get<reg::RB_DEPTH_INFO>();
    DepthTarget* d = GetDepthTarget(depth_info.depth_base, width, height, true);
    if (d) {
      d->last_used = frame_;
      key.depth = d->texture;
      min_w = std::min(min_w, d->width);
      min_h = std::min(min_h, d->height);
    }
  }
  if (!used_colors && !key.depth) {
    return false;
  }
  GLuint fbo = GetFramebuffer(key);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  bound_width_ = min_w;
  bound_height_ = min_h;
  bound_color_mask_ = used_colors;
  bound_depth_ = key.depth != 0;
  return true;
}

bool GlRenderTargets::Resolve(uint32_t& extent_start_out, uint32_t& extent_length_out) {
  extent_start_out = 0;
  extent_length_out = 0;
  draw_util::ResolveInfo ri;
  if (!draw_util::GetResolveInfo(regs_, memory_, 1, 1, false, false, ri)) {
    return false;
  }
  auto surface_info = regs_.Get<reg::RB_SURFACE_INFO>();
  bool is_depth = ri.IsCopyingDepth();
  uint32_t log2x = surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 1 : 0;
  uint32_t log2y = surface_info.msaa_samples >= xenos::MsaaSamples::k2X ? 1 : 0;
  uint32_t pitch_tiles = xenos::GetSurfacePitchTiles(surface_info.surface_pitch, surface_info.msaa_samples, false);
  ++resolve_count_;

  // Pixel rectangle of the resolve within the render target.
  uint32_t src_original_base = is_depth ? ri.depth_original_base : ri.color_original_base;
  const draw_util::ResolveEdramInfo& edram_info = is_depth ? ri.depth_edram_info : ri.color_edram_info;
  uint32_t base_offset_tiles =
      ((edram_info.base_tiles - src_original_base) & 0x7FF) >> (edram_info.format_is_64bpp ? 1 : 0);
  uint32_t base_off_x_tiles = pitch_tiles ? base_offset_tiles % pitch_tiles : 0;
  uint32_t base_off_y_tiles = pitch_tiles ? base_offset_tiles / pitch_tiles : 0;
  uint32_t x0 = ((base_off_x_tiles * xenos::kEdramTileWidthSamples) >> log2x) +
                ri.coordinate_info.edram_offset_x_div_8 * 8;
  uint32_t y0 = ((base_off_y_tiles * xenos::kEdramTileHeightSamples) >> log2y) +
                ri.coordinate_info.edram_offset_y_div_8 * 8;
  uint32_t w = ri.coordinate_info.width_div_8 * 8;
  uint32_t h = ri.height_div_8 * 8;

  // ---- Copy ----
  if (!is_depth && ri.copy_dest_extent_length) {
    auto color_info = regs_.Get<reg::RB_COLOR_INFO>(
        reg::RB_COLOR_INFO::rt_register_indices[ri.rb_copy_control.copy_src_select]);
    ColorTarget* rt = GetColorTarget(color_info.color_base, color_info.color_format, x0 + w, y0 + h, true);
    if (rt) {
      uint32_t rw = std::min(w, rt->width > x0 ? rt->width - x0 : 0);
      uint32_t rh = std::min(h, rt->height > y0 ? rt->height - y0 : 0);
      xenos::TextureFormat dest_format = xenos::TextureFormat(ri.copy_dest_info.copy_dest_format);
      const FormatInfo* fi = FormatInfo::Get(dest_format);
      uint32_t bpp = fi ? fi->bits_per_pixel / 8 : 4;
      uint32_t bpp_log2 = bpp >= 16 ? 4 : bpp >= 8 ? 3 : bpp >= 4 ? 2 : bpp >= 2 ? 1 : 0;
      auto dest_pitch_reg = regs_.Get<reg::RB_COPY_DEST_PITCH>();
      uint32_t dest_pitch = dest_pitch_reg.copy_dest_pitch;
      uint32_t ox = ri.copy_dest_coordinate_info.offset_x_div_8 * 8;
      uint32_t oy = ri.copy_dest_coordinate_info.offset_y_div_8 * 8;
      xenos::Endian128 endian = ri.copy_dest_info.copy_dest_endian;
      bool swap_rb = ri.copy_dest_info.copy_dest_swap;
      bool src_is_8888 = rt->internal_format == GL_RGBA8;
      if (rw && rh) {
        std::vector<uint8_t> px8;
        std::vector<float> pxf;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo_);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->texture, 0);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        {
          perf::Scope perf_readback_scope(perf::kReadbackTicks);
          if (src_is_8888) {
            px8.resize(size_t(rw) * rh * 4);
            glReadPixels(x0, y0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, px8.data());
          } else {
            pxf.resize(size_t(rw) * rh * 4);
            glReadPixels(x0, y0, rw, rh, GL_RGBA, GL_FLOAT, pxf.data());
          }
        }
        const bool log_first = resolve_count_ <= 4;
        uint8_t* dest_base = memory_.TranslatePhysical(ri.copy_dest_base);
        uint8_t texel[16];
        for (uint32_t j = 0; j < rh; ++j) {
          for (uint32_t i = 0; i < rw; ++i) {
            float c[4];
            uint8_t b8[4];
            size_t si = size_t(j) * rw + i;
            if (src_is_8888) {
              for (int k = 0; k < 4; ++k) {
                b8[k] = px8[si * 4 + k];
                c[k] = b8[k] * (1.0f / 255.0f);
              }
            } else {
              for (int k = 0; k < 4; ++k) {
                c[k] = pxf[si * 4 + k];
                b8[k] = uint8_t(std::clamp(c[k], 0.0f, 1.0f) * 255.0f + 0.5f);
              }
            }
            if (swap_rb) {
              std::swap(b8[0], b8[2]);
              std::swap(c[0], c[2]);
            }
            std::memset(texel, 0, sizeof(texel));
            switch (xenos::ColorFormat(dest_format)) {
              case xenos::ColorFormat::k_8_8_8_8:
              case xenos::ColorFormat::k_8_8_8_8_A:
              case xenos::ColorFormat::k_8_8_8_8_AS_16_16_16_16:
                std::memcpy(texel, b8, 4);
                break;
              case xenos::ColorFormat::k_8:
              case xenos::ColorFormat::k_8_A:
              case xenos::ColorFormat::k_8_B:
                texel[0] = b8[0];
                break;
              case xenos::ColorFormat::k_8_8:
                texel[0] = b8[0];
                texel[1] = b8[1];
                break;
              case xenos::ColorFormat::k_5_6_5: {
                uint16_t v = uint16_t(((b8[0] >> 3) << 11) | ((b8[1] >> 2) << 5) | (b8[2] >> 3));
                std::memcpy(texel, &v, 2);
                break;
              }
              case xenos::ColorFormat::k_1_5_5_5: {
                uint16_t v = uint16_t(((b8[3] >> 7) << 15) | ((b8[0] >> 3) << 10) |
                                      ((b8[1] >> 3) << 5) | (b8[2] >> 3));
                std::memcpy(texel, &v, 2);
                break;
              }
              case xenos::ColorFormat::k_4_4_4_4: {
                uint16_t v = uint16_t(((b8[3] >> 4) << 12) | ((b8[0] >> 4) << 8) | ((b8[1] >> 4) << 4) |
                                      (b8[2] >> 4));
                std::memcpy(texel, &v, 2);
                break;
              }
              case xenos::ColorFormat::k_2_10_10_10:
              case xenos::ColorFormat::k_2_10_10_10_AS_16_16_16_16: {
                uint32_t r = uint32_t(std::clamp(c[0], 0.0f, 1.0f) * 1023.0f + 0.5f);
                uint32_t g = uint32_t(std::clamp(c[1], 0.0f, 1.0f) * 1023.0f + 0.5f);
                uint32_t bl = uint32_t(std::clamp(c[2], 0.0f, 1.0f) * 1023.0f + 0.5f);
                uint32_t a = uint32_t(std::clamp(c[3], 0.0f, 1.0f) * 3.0f + 0.5f);
                uint32_t v = r | (g << 10) | (bl << 20) | (a << 30);
                std::memcpy(texel, &v, 4);
                break;
              }
              case xenos::ColorFormat::k_16_16_16_16_FLOAT:
              case xenos::ColorFormat::k_16_16_FLOAT:
              case xenos::ColorFormat::k_16_FLOAT: {
                uint16_t hv[4] = {FloatToHalf(c[0]), FloatToHalf(c[1]), FloatToHalf(c[2]), FloatToHalf(c[3])};
                std::memcpy(texel, hv, bpp);
                break;
              }
              case xenos::ColorFormat::k_32_FLOAT:
                std::memcpy(texel, &c[0], 4);
                break;
              case xenos::ColorFormat::k_32_32_FLOAT:
                std::memcpy(texel, &c[0], 4);
                std::memcpy(texel + 4, &c[1], 4);
                break;
              case xenos::ColorFormat::k_32_32_32_32_FLOAT:
                std::memcpy(texel, c, 16);
                break;
              case xenos::ColorFormat::k_16:
              case xenos::ColorFormat::k_16_16:
              case xenos::ColorFormat::k_16_16_16_16: {
                uint16_t uv[4];
                for (int k = 0; k < 4; ++k) {
                  uv[k] = uint16_t(std::clamp(c[k], 0.0f, 1.0f) * 65535.0f + 0.5f);
                }
                std::memcpy(texel, uv, bpp);
                break;
              }
              default:
                std::memcpy(texel, b8, 4);
                break;
            }
            ApplyEndian(texel, bpp, endian);
            uint32_t off = uint32_t(texture_util::GetTiledOffset2D(int32_t(ox + i), int32_t(oy + j),
                                                                   dest_pitch, bpp_log2));
            std::memcpy(dest_base + off, texel, bpp);
          }
        }
        if (log_first) {
          REXGPU_INFO(
              "GL resolve #{}: RT base={} {}x{} fmt={} -> dest 0x{:08X} fmt={} pitch={} rect=({},{}) {}x{} "
              "endian={} swap={}",
              resolve_count_, uint32_t(color_info.color_base), rt->width, rt->height, uint32_t(color_info.color_format),
              ri.copy_dest_base, uint32_t(dest_format), dest_pitch, x0, y0, rw, rh, uint32_t(endian),
              uint32_t(swap_rb));
        }
        extent_start_out = ri.copy_dest_extent_start;
        extent_length_out = ri.copy_dest_extent_length;
      }
    }
  } else if (is_depth) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXGPU_WARN("GL resolve: depth copy to memory is not implemented yet (skipped)");
    }
  }

  // ---- Clear ----
  if (ri.IsClearingColor()) {
    auto color_info = regs_.Get<reg::RB_COLOR_INFO>(
        reg::RB_COLOR_INFO::rt_register_indices[ri.rb_copy_control.copy_src_select]);
    ColorTarget* rt = GetColorTarget(color_info.color_base, color_info.color_format, x0 + w, y0 + h, true);
    if (rt) {
      float c[4];
      using F = xenos::ColorRenderTargetFormat;
      uint32_t v0 = ri.rb_color_clear, v1 = ri.rb_color_clear_lo;
      switch (color_info.color_format) {
        case F::k_8_8_8_8:
        case F::k_8_8_8_8_GAMMA:
          for (int k = 0; k < 4; ++k) {
            c[k] = float((v0 >> (8 * k)) & 0xFF) * (1.0f / 255.0f);
          }
          break;
        case F::k_2_10_10_10:
        case F::k_2_10_10_10_AS_10_10_10_10:
          c[0] = float(v0 & 0x3FF) * (1.0f / 1023.0f);
          c[1] = float((v0 >> 10) & 0x3FF) * (1.0f / 1023.0f);
          c[2] = float((v0 >> 20) & 0x3FF) * (1.0f / 1023.0f);
          c[3] = float(v0 >> 30) * (1.0f / 3.0f);
          break;
        case F::k_16_16_FLOAT:
        case F::k_16_16_16_16_FLOAT:
          c[0] = HalfToFloat(uint16_t(v0));
          c[1] = HalfToFloat(uint16_t(v0 >> 16));
          c[2] = HalfToFloat(uint16_t(v1));
          c[3] = HalfToFloat(uint16_t(v1 >> 16));
          break;
        case F::k_32_FLOAT:
        case F::k_32_32_FLOAT:
          std::memcpy(&c[0], &v0, 4);
          std::memcpy(&c[1], &v1, 4);
          c[2] = c[3] = 0.0f;
          break;
        default:
          c[0] = c[1] = c[2] = c[3] = 0.0f;
          break;
      }
      GLuint fbo = GetFramebuffer(FbKey{{rt->texture, 0, 0, 0}, 0});
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glEnable(GL_SCISSOR_TEST);
      glScissor(x0, y0, w, h);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      glClearBufferfv(GL_COLOR, 0, c);
      glDisable(GL_SCISSOR_TEST);
    }
  }
  if (ri.IsClearingDepth()) {
    auto depth_info = regs_.Get<reg::RB_DEPTH_INFO>();
    DepthTarget* d = GetDepthTarget(depth_info.depth_base, x0 + w, y0 + h, true);
    if (d) {
      uint32_t v = ri.rb_depth_clear;
      float depth;
      if (depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
        depth = std::min(xenos::Float20e4To32(v >> 8), 1.0f);
      } else {
        depth = xenos::UNorm24To32(v >> 8);
      }
      GLint stencil = GLint(v & 0xFF);
      GLuint fbo = GetFramebuffer(FbKey{{0, 0, 0, 0}, d->texture});
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glEnable(GL_SCISSOR_TEST);
      glScissor(x0, y0, w, h);
      glDepthMask(GL_TRUE);
      glStencilMask(0xFF);
      glClearBufferfi(GL_DEPTH_STENCIL, 0, depth, stencil);
      glDisable(GL_SCISSOR_TEST);
    }
  }
  return true;
}

}  // namespace rex::graphics::gl

namespace rex::graphics::gl {
bool GlRenderTargets::DebugReadColor(uint32_t base, std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h) {
  for (auto& kv : color_targets_) {
    if (kv.second.base != base) continue;
    ColorTarget& t = kv.second;
    w = t.width;
    h = t.height;
    rgba.resize(size_t(w) * h * 4);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo_);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.texture, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    return true;
  }
  return false;
}
}  // namespace rex::graphics::gl
