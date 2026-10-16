#include "render_targets.h"
#include "perf.h"

#include <algorithm>
#include <cstring>

#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/math.h>

#include "gl_cvars.h"
#include "gl_ext.h"
#include "shared_memory.h"
#include "spirv_to_glsl.h"
#include "tiling.h"

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
  if (REXCVAR_GET(gl_gpu_resolve)) {
    if (!CreateGpuResolveProgram()) {
      REXGPU_WARN("GL RT: GPU resolve compute shader unavailable, using CPU resolves");
    }
  }
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

bool GlRenderTargets::Resolve(uint32_t& extent_start_out, uint32_t& extent_length_out, bool& gpu_written_out) {
  gpu_written_out = false;
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
  gpu_written_out = false;
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
        PendingResolve pr;
        pr.frame = frame_;
        pr.dest_base = ri.copy_dest_base;
        pr.extent_start = ri.copy_dest_extent_start;
        pr.extent_length = ri.copy_dest_extent_length;
        pr.ox = ox;
        pr.oy = oy;
        pr.rw = rw;
        pr.rh = rh;
        pr.dest_pitch = dest_pitch;
        pr.dest_format = dest_format;
        pr.bpp = bpp;
        pr.bpp_log2 = bpp_log2;
        pr.endian = endian;
        pr.swap_rb = swap_rb;
        pr.src_is_8888 = src_is_8888;
        const bool sync = REXCVAR_GET(gl_sync_resolve);
        // Snapshot kind: the dest layout equals the render target's host format, so consumers can sample a GPU
        // copy of the render target instead of going through guest memory.
        {
          const bool rgba8_dest = rt->internal_format == GL_RGBA8 && bpp == 4 &&
                                  (dest_format == xenos::TextureFormat::k_8_8_8_8 ||
                                   dest_format == xenos::TextureFormat::k_8_8_8_8_A);
          const bool f16_dest = rt->internal_format == GL_RGBA16F && !swap_rb &&
                                dest_format == xenos::TextureFormat::k_16_16_16_16_FLOAT;
          const bool r10_dest = rt->internal_format == GL_RGB10_A2 && bpp == 4 && !swap_rb &&
                                dest_format == xenos::TextureFormat::k_2_10_10_10;
          if (!sync && ox == 0 && oy == 0 && REXCVAR_GET(gl_resolve_gpu_copy) &&
              uint64_t(rw) * rh <= (1u << 21) && (rgba8_dest || f16_dest || r10_dest)) {
            pr.snap_kind = rgba8_dest ? 1 : f16_dest ? 2 : 3;
            pr.snapshot_format = rt->internal_format;
          }
        }
        bool gpu_done = false;
        if (!sync && gpu_prog_ && sm_ && REXCVAR_GET(gl_gpu_resolve)) {
          gpu_done = GpuResolve(*rt, pr, x0, y0);
        }
        if (!gpu_done) {
          perf::Add(perf::kCpuResolves);
          // CPU path: pending resolves into the same range must be written back first (ordering), and the range
          // is no longer GPU resident.
          if (!pending_.empty()) {
            FlushPending(ri.copy_dest_extent_start, ri.copy_dest_extent_length, FlushReason::kOther);
          }
          ResidentRemove(ri.copy_dest_extent_start, ri.copy_dest_extent_length);
          pr.pbo_size = rw * rh * (src_is_8888 ? 4 : 16);
          {
            perf::Scope perf_readback_scope(perf::kReadbackTicks);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo_);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->texture, 0);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            for (size_t k = 0; k < free_pbos_.size(); ++k) {
              if (free_pbos_[k].second >= pr.pbo_size && free_pbos_[k].second <= pr.pbo_size * 2) {
                pr.pbo = free_pbos_[k].first;
                pr.pbo_size = free_pbos_[k].second;
                free_pbos_.erase(free_pbos_.begin() + k);
                break;
              }
            }
            if (!pr.pbo) {
              glGenBuffers(1, &pr.pbo);
              glBindBuffer(GL_PIXEL_PACK_BUFFER, pr.pbo);
              glBufferData(GL_PIXEL_PACK_BUFFER, pr.pbo_size, nullptr, GL_STREAM_READ);
            } else {
              glBindBuffer(GL_PIXEL_PACK_BUFFER, pr.pbo);
            }
            if (src_is_8888) {
              glReadPixels(x0, y0, rw, rh, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            } else {
              glReadPixels(x0, y0, rw, rh, GL_RGBA, GL_FLOAT, nullptr);
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            pr.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
          }
        }
        if (pr.snap_kind) {
          // GPU-side snapshot of the resolved rectangle for consumers (textures).
          for (size_t k = 0; k < free_snapshots_.size(); ++k) {
            if (free_snapshots_[k].w == rw && free_snapshots_[k].h == rh &&
                free_snapshots_[k].format == pr.snapshot_format) {
              pr.snapshot = free_snapshots_[k].texture;
              free_snapshots_.erase(free_snapshots_.begin() + k);
              break;
            }
          }
          if (!pr.snapshot) {
            glGenTextures(1, &pr.snapshot);
            glBindTexture(GL_TEXTURE_2D_ARRAY, pr.snapshot);
            glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, pr.snapshot_format, rw, rh, 1);
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
          }
          pr.snapshot_w = rw;
          pr.snapshot_h = rh;
          const int gpu_snap_begin = perf::GpuStamp();
          glCopyImageSubData(rt->texture, GL_TEXTURE_2D, 0, GLint(x0), GLint(y0), 0, pr.snapshot,
                             GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, GLsizei(rw), GLsizei(rh), 1);
          perf::GpuSegEnd(perf::kGpuSegSnapshot, gpu_snap_begin);
        }
        // Log distinct resolve configurations (up to gl_resolve_log * 4) for diagnostics.
        {
          const int32_t log_budget = REXCVAR_GET(gl_resolve_log);
          uint64_t sig = (uint64_t(pr.dest_base) << 32) ^ (uint64_t(rw) << 16) ^ rh ^ (uint64_t(dest_pitch) << 48) ^
                         (uint64_t(uint32_t(dest_format)) << 8) ^ (uint64_t(uint32_t(endian)) << 4) ^
                         (swap_rb ? 1 : 0) ^ (uint64_t(ox + oy * 4096) << 20);
          bool seen = false;
          for (uint64_t sg : resolve_sigs_) seen |= sg == sig;
          if (!seen && int32_t(resolve_sigs_.size()) < log_budget * 4) {
            resolve_sigs_.push_back(sig);
            REXGPU_INFO(
                "GL resolve #{} (new config): RT base={} {}x{} fmt={} -> dest 0x{:08X} fmt={} pitch={} (aligned {}) "
                "origin=({},{}) src rect=({},{}) {}x{} extent=0x{:08X}+0x{:X} endian={} swap={} gpu={} snap_kind={}",
                resolve_count_, uint32_t(color_info.color_base), rt->width, rt->height,
                uint32_t(color_info.color_format), ri.copy_dest_base, uint32_t(dest_format), dest_pitch,
                (dest_pitch + 31) & ~31u, ox, oy, x0, y0, rw, rh, ri.copy_dest_extent_start,
                ri.copy_dest_extent_length, uint32_t(endian), uint32_t(swap_rb), gpu_done, pr.snap_kind);
          }
        }
        pending_.push_back(pr);
        if (sync) {
          FlushAll(FlushReason::kSync);
        } else {
          while (pending_.size() > 24) FlushFront(true, FlushReason::kCap);
        }
        extent_start_out = ri.copy_dest_extent_start;
        extent_length_out = ri.copy_dest_extent_length;
        gpu_written_out = gpu_done;
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

namespace rex::graphics::gl {

void GlRenderTargets::ConvertPending(PendingResolve& pr, const void* pixels) {
  perf::Scope perf_conv_scope(perf::kResolveConvTicks);
  const uint32_t rw = pr.rw, rh = pr.rh, ox = pr.ox, oy = pr.oy, dest_pitch = pr.dest_pitch, bpp = pr.bpp,
                 bpp_log2 = pr.bpp_log2;
  const bool src_is_8888 = pr.src_is_8888, swap_rb = pr.swap_rb;
  const xenos::TextureFormat dest_format = pr.dest_format;
  const xenos::Endian128 endian = pr.endian;
  uint8_t* dest_base = memory_.TranslatePhysical(pr.dest_base);
  const uint8_t* px8 = src_is_8888 ? static_cast<const uint8_t*>(pixels) : nullptr;
  const float* pxf = src_is_8888 ? nullptr : static_cast<const float*>(pixels);
  const bool fast_dest = dest_format == xenos::TextureFormat::k_8_8_8_8 ||
                         dest_format == xenos::TextureFormat::k_8_8_8_8_A ||
                         dest_format == xenos::TextureFormat::k_8_8_8_8_AS_16_16_16_16;
  if (src_is_8888 && bpp == 4 && fast_dest && uint32_t(endian) <= uint32_t(xenos::Endian128::k16in32)) {
    // Fast path: 32-bit texel, endian swap by bit tricks, inline tiling.
    const uint32_t pitch_aligned = (dest_pitch + 31) & ~31u;
    for (uint32_t j = 0; j < rh; ++j) {
      const uint8_t* row = px8 + size_t(j) * rw * 4;
      for (uint32_t i = 0; i < rw; ++i) {
        uint32_t v;
        std::memcpy(&v, row + size_t(i) * 4, 4);
        if (swap_rb) v = (v & 0xFF00FF00u) | ((v & 0xFFu) << 16) | ((v >> 16) & 0xFFu);
        switch (endian) {
          case xenos::Endian128::k8in16: v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu); break;
          case xenos::Endian128::k8in32: v = __builtin_bswap32(v); break;
          case xenos::Endian128::k16in32: v = (v << 16) | (v >> 16); break;
          default: break;
        }
        std::memcpy(dest_base + TiledOffset2D(ox + i, oy + j, pitch_aligned, 2), &v, 4);
      }
    }
    return;
  }
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
}

void GlRenderTargets::ReleaseDownload(PendingResolve& pr) {
  if (pr.fence) {
    glDeleteSync(pr.fence);
    pr.fence = nullptr;
  }
  if (pr.pbo) {
    free_pbos_.emplace_back(pr.pbo, pr.pbo_size);
    pr.pbo = 0;
    while (free_pbos_.size() > 6) {
      glDeleteBuffers(1, &free_pbos_.front().first);
      free_pbos_.erase(free_pbos_.begin());
    }
  }
}

void GlRenderTargets::ReleaseSnapshot(PendingResolve& pr) {
  if (pr.snapshot) {
    free_snapshots_.push_back({pr.snapshot, pr.snapshot_w, pr.snapshot_h, pr.snapshot_format});
    pr.snapshot = 0;
    while (free_snapshots_.size() > 6) {
      glDeleteTextures(1, &free_snapshots_.front().texture);
      free_snapshots_.erase(free_snapshots_.begin());
    }
  }
}

void GlRenderTargets::ReleasePending(PendingResolve& pr) {
  ReleaseDownload(pr);
  ReleaseSnapshot(pr);
}

// ---------------------------------------------------------------------------------------------
// GPU resolve (compute shader writing the guest-format image into the shared memory store)
// ---------------------------------------------------------------------------------------------

namespace {

// Dest layout kinds of the compute shader.
constexpr int kGpuKind8888 = 0;      // 4x8 bit
constexpr int kGpuKind2101010 = 1;   // 2_10_10_10
constexpr int kGpuKind1616F = 2;     // 16_16_FLOAT (32 bpp)
constexpr int kGpuKind16161616F = 3; // 16_16_16_16_FLOAT (64 bpp)
constexpr int kGpuKind32F = 4;       // 32_FLOAT
constexpr int kGpuKind3232F = 5;     // 32_32_FLOAT (64 bpp)

// Body of the resolve compute shader; prefixed with "#version" and optionally "#define SRC_ARRAY" (source is a
// GL_TEXTURE_2D_ARRAY resolve snapshot, layer 0, instead of the 2D render target).
const char* kResolveCs = R"(
layout(local_size_x = 8, local_size_y = 8) in;
#ifdef SRC_ARRAY
layout(binding = 0) uniform sampler2DArray src;
#define SRC_FETCH(q) texelFetch(src, ivec3(q, 0), 0)
#else
layout(binding = 0) uniform sampler2D src;
#define SRC_FETCH(q) texelFetch(src, q, 0)
#endif
layout(std430, binding = 0) buffer StoreBuf { uint data[]; } sm;
layout(std430, binding = 1) readonly buffer PtBuf { uint pt[]; };
uniform ivec4 u_src;   // source rect x0, y0, width, height
uniform ivec4 u_dst;   // dest origin x, y (texels), pitch aligned to 32, log2 bytes per texel
uniform uint u_base;   // guest physical base address of the destination image
uniform ivec4 u_fmt;   // dest kind, endian (0..3), swap red/blue, unused

uint tiled(uint x, uint y, uint pitch, uint bl) {
  int macro = int((x >> 5) + (y >> 5) * (pitch >> 5)) << int(bl + 7u);
  int micro = int((x & 7u) + ((y & 0xEu) << 2)) << int(bl);
  int offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + int((y & 1u) << 4);
  return uint(((offset & ~0x1FF) << 3) + int((y & 16u) << 7) + ((offset & 0x1C0) << 2) +
              int(((((y & 8u) >> 2) + (x >> 3)) & 3u) << 6) + (offset & 0x3F));
}

uint swapw(uint v, int e) {
  if (e == 1) return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
  if (e == 2) return (v << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
  if (e == 3) return (v << 16) | (v >> 16);
  return v;
}

void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_src.z || p.y >= u_src.w) return;
  vec4 c = SRC_FETCH(u_src.xy + p);
  if (u_fmt.z != 0) c = c.bgra;
  uint addr = u_base + tiled(uint(u_dst.x + p.x), uint(u_dst.y + p.y), uint(u_dst.z), uint(u_dst.w));
  uint slot = pt[addr >> 20];
  if (slot == 0xFFFFFFFFu) return;
  uint idx = slot + ((addr & 0xFFFFFu) >> 2);
  int kind = u_fmt.x;
  int e = u_fmt.y;
  uint v0 = 0u, v1 = 0u;
  if (kind == 0) {
    uvec4 b = uvec4(clamp(c, 0.0, 1.0) * 255.0 + 0.5);
    v0 = b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
  } else if (kind == 1) {
    vec4 q = clamp(c, 0.0, 1.0);
    uint r = uint(q.r * 1023.0 + 0.5), g = uint(q.g * 1023.0 + 0.5), bl = uint(q.b * 1023.0 + 0.5);
    uint a = uint(q.a * 3.0 + 0.5);
    v0 = r | (g << 10) | (bl << 20) | (a << 30);
  } else if (kind == 2) {
    v0 = packHalf2x16(c.rg);
  } else if (kind == 3) {
    v0 = packHalf2x16(c.rg);
    v1 = packHalf2x16(c.ba);
  } else if (kind == 4) {
    v0 = floatBitsToUint(c.r);
  } else {
    v0 = floatBitsToUint(c.r);
    v1 = floatBitsToUint(c.g);
  }
  sm.data[idx] = swapw(v0, e);
  if (kind == 3 || kind == 5) sm.data[idx + 1u] = swapw(v1, e);
}
)";

int GpuKindFor(xenos::TextureFormat dest_format, uint32_t bpp) {
  int kind = -1;
  switch (xenos::ColorFormat(dest_format)) {
    case xenos::ColorFormat::k_8_8_8_8:
    case xenos::ColorFormat::k_8_8_8_8_A:
    case xenos::ColorFormat::k_8_8_8_8_AS_16_16_16_16:
      kind = kGpuKind8888;
      break;
    case xenos::ColorFormat::k_2_10_10_10:
    case xenos::ColorFormat::k_2_10_10_10_AS_16_16_16_16:
      kind = kGpuKind2101010;
      break;
    case xenos::ColorFormat::k_16_16_FLOAT:
      kind = kGpuKind1616F;
      break;
    case xenos::ColorFormat::k_16_16_16_16_FLOAT:
      kind = kGpuKind16161616F;
      break;
    case xenos::ColorFormat::k_32_FLOAT:
      kind = kGpuKind32F;
      break;
    case xenos::ColorFormat::k_32_32_FLOAT:
      kind = kGpuKind3232F;
      break;
    default:
      return -1;
  }
  const uint32_t want_bpp = (kind == kGpuKind16161616F || kind == kGpuKind3232F) ? 8 : 4;
  return bpp == want_bpp ? kind : -1;
}

}  // namespace

namespace {
GLuint CompileResolveProgram(bool src_array) {
  GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
  const char* parts[2] = {src_array ? "#version 430 core\n#define SRC_ARRAY 1\n" : "#version 430 core\n", kResolveCs};
  glShaderSource(cs, 2, parts, nullptr);
  glCompileShader(cs);
  GLint ok = 0;
  glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(cs, sizeof(log), nullptr, log);
    REXGPU_ERROR("GL RT: resolve compute shader (array={}) compile failed: {}", src_array, log);
    glDeleteShader(cs);
    return 0;
  }
  GLuint prog = glCreateProgram();
  glAttachShader(prog, cs);
  glLinkProgram(prog);
  glDeleteShader(cs);
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
    REXGPU_ERROR("GL RT: resolve compute shader (array={}) link failed: {}", src_array, log);
    glDeleteProgram(prog);
    return 0;
  }
  return prog;
}
}  // namespace

bool GlRenderTargets::CreateGpuResolveProgram() {
  GLuint prog = CompileResolveProgram(false);
  if (!prog) return false;
  gpu_prog_ = prog;
  u_src_ = glGetUniformLocation(prog, "u_src");
  u_dst_ = glGetUniformLocation(prog, "u_dst");
  u_base_ = glGetUniformLocation(prog, "u_base");
  u_fmt_ = glGetUniformLocation(prog, "u_fmt");
  gpu_prog_array_ = CompileResolveProgram(true);
  if (gpu_prog_array_) {
    ua_src_ = glGetUniformLocation(gpu_prog_array_, "u_src");
    ua_dst_ = glGetUniformLocation(gpu_prog_array_, "u_dst");
    ua_base_ = glGetUniformLocation(gpu_prog_array_, "u_base");
    ua_fmt_ = glGetUniformLocation(gpu_prog_array_, "u_fmt");
  }
  REXGPU_INFO("GL RT: GPU resolve compute shader ready (snapshot variant {}, lazy resolves {})",
              gpu_prog_array_ ? "ready" : "FAILED", gpu_prog_array_ && REXCVAR_GET(gl_lazy_resolve) ? "on" : "off");
  return true;
}

bool GlRenderTargets::GpuResolve(ColorTarget& rt, PendingResolve& pr, uint32_t x0, uint32_t y0) {
  const int kind = GpuKindFor(pr.dest_format, pr.bpp);
  if (kind < 0 || uint32_t(pr.endian) > uint32_t(xenos::Endian128::k16in32)) {
    return false;
  }
  if (uint64_t(pr.extent_start) + pr.extent_length > SharedMemory::kBufferSize) {
    return false;
  }
  perf::Scope perf_gpu_scope(perf::kGpuResolveTicks);
  // Make the destination pages resident (and current: partially covered edge pages keep their guest contents).
  if (!sm_->RequestRange(pr.extent_start, pr.extent_length)) {
    return false;
  }
  if (!sm_->store_buffer()) {
    return false;
  }
  pr.gpu_kind = kind;
  const bool lazy = pr.snap_kind && gpu_prog_array_ && REXCVAR_GET(gl_lazy_resolve);
  const uint32_t new_pitch = (pr.dest_pitch + 31) & ~31u;
  // Older resolves whose image is completely overwritten by this one never need to be written back (or, when
  // lazy, never written into the store at all). Older lazy ones that overlap only partially must reach the store
  // before this one does.
  const uint64_t ns = pr.extent_start, ne = ns + pr.extent_length;
  for (PendingResolve& o : pending_) {
    if (!o.gpu || o.done || o.dl_started) continue;
    const bool covered = o.dest_base == pr.dest_base && ((o.dest_pitch + 31) & ~31u) == new_pitch &&
                         o.bpp == pr.bpp && o.ox >= pr.ox && o.oy >= pr.oy && o.ox + o.rw <= pr.ox + pr.rw &&
                         o.oy + o.rh <= pr.oy + pr.rh;
    if (covered) {
      if (o.lazy) {
        o.lazy = false;
        --lazy_count_;
      }
      o.done = true;
      ReleaseSnapshot(o);
      perf::Add(perf::kResolveSuperseded);
    } else if (o.lazy && uint64_t(o.extent_start) < ne && uint64_t(o.extent_start) + o.extent_length > ns) {
      Materialize(o);
    }
  }
  if (!lazy) {
    GpuResolveDispatch(pr, rt.texture, false, x0, y0);
    if (!gpu_prog_) return false;  // GL error on the first dispatches: CPU fallback
  }
  pr.gpu = true;
  if (lazy) {
    // The snapshot (made by the caller right after this) holds the image; the store is written on demand.
    pr.lazy = true;
    ++lazy_count_;
    perf::Add(perf::kLazyDeferred);
  }
  perf::Add(perf::kGpuResolves);
  ResidentAdd(pr.extent_start, pr.extent_length);
  return true;
}

void GlRenderTargets::GpuResolveDispatch(PendingResolve& pr, GLuint texture, bool from_snapshot, uint32_t x0,
                                         uint32_t y0) {
  const uint32_t pitch_aligned = (pr.dest_pitch + 31) & ~31u;
  if (gpu_checks_left_ > 0) {
    while (glGetError() != GL_NO_ERROR) {}  // so only errors of the dispatch below are blamed on it
  }
  glUseProgram(from_snapshot ? gpu_prog_array_ : gpu_prog_);
  glActiveTexture(GL_TEXTURE0);
  const GLenum target = from_snapshot ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
  glBindTexture(target, texture);
  glBindSampler(0, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sm_->store_buffer());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sm_->page_table_buffer());
  glUniform4i(from_snapshot ? ua_src_ : u_src_, GLint(x0), GLint(y0), GLint(pr.rw), GLint(pr.rh));
  glUniform4i(from_snapshot ? ua_dst_ : u_dst_, GLint(pr.ox), GLint(pr.oy), GLint(pitch_aligned),
              GLint(pr.bpp_log2));
  glUniform1ui(from_snapshot ? ua_base_ : u_base_, pr.dest_base);
  glUniform4i(from_snapshot ? ua_fmt_ : u_fmt_, pr.gpu_kind, GLint(uint32_t(pr.endian)), pr.swap_rb ? 1 : 0, 0);
  const int gpu_cs_begin = perf::GpuStamp();
  glDispatchCompute((pr.rw + 7) / 8, (pr.rh + 7) / 8, 1);
  perf::GpuSegEnd(perf::kGpuSegResolve, gpu_cs_begin);
  glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT |
                  GL_ELEMENT_ARRAY_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT |
                  GL_UNIFORM_BARRIER_BIT);
  glBindTexture(target, 0);
  glUseProgram(0);
  if (gpu_checks_left_ > 0) {
    --gpu_checks_left_;
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
      REXGPU_ERROR("GL RT: GPU resolve (snapshot={}) raised GL error 0x{:x}, falling back to CPU resolves",
                   from_snapshot, err);
      glDeleteProgram(gpu_prog_);
      gpu_prog_ = 0;
      if (gpu_prog_array_) glDeleteProgram(gpu_prog_array_);
      gpu_prog_array_ = 0;
    }
  }
}

void GlRenderTargets::Materialize(PendingResolve& pr) {
  if (!pr.lazy) return;
  pr.lazy = false;
  --lazy_count_;
  perf::Add(perf::kLazyMaterialized);
  if (!pr.snapshot || !gpu_prog_array_) {
    REXGPU_WARN("GL RT: lazy resolve 0x{:08X} lost its snapshot, store left stale", pr.dest_base);
    return;
  }
  perf::Scope perf_gpu_scope(perf::kGpuResolveTicks);
  GpuResolveDispatch(pr, pr.snapshot, true, 0, 0);
}

void GlRenderTargets::MaterializeLazySlow(uint32_t start, uint32_t length) {
  const uint64_t s = start, e = s + length;
  // Oldest first: the store must see the writes in resolve order.
  for (PendingResolve& pr : pending_) {
    if (pr.lazy && !pr.done && uint64_t(pr.extent_start) < e && uint64_t(pr.extent_start) + pr.extent_length > s) {
      Materialize(pr);
    }
  }
}

void GlRenderTargets::StartGpuDownload(PendingResolve& pr) {
  if (pr.lazy) {
    // Store writes keep resolve order: older lazy ones overlapping this range first (pr itself included).
    MaterializeLazySlow(pr.extent_start, pr.extent_length);
  }
  pr.dl_started = true;
  const uint32_t size = pr.extent_length;
  pr.pbo_size = size;
  for (size_t k = 0; k < free_pbos_.size(); ++k) {
    if (free_pbos_[k].second >= size && free_pbos_[k].second <= size * 2) {
      pr.pbo = free_pbos_[k].first;
      pr.pbo_size = free_pbos_[k].second;
      free_pbos_.erase(free_pbos_.begin() + k);
      break;
    }
  }
  if (!pr.pbo) {
    glGenBuffers(1, &pr.pbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, pr.pbo);
    glBufferData(GL_COPY_WRITE_BUFFER, pr.pbo_size, nullptr, GL_STREAM_READ);
  } else {
    glBindBuffer(GL_COPY_WRITE_BUFFER, pr.pbo);
  }
  glBindBuffer(GL_COPY_READ_BUFFER, sm_->store_buffer());
  uint32_t copied = 0;
  bool ok = true;
  while (copied < size) {
    GLintptr off = 0;
    uint32_t n = 0;
    if (!sm_->StoreOffset(pr.extent_start + copied, off, n)) {
      ok = false;
      break;
    }
    n = std::min(n, size - copied);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, off, GLintptr(copied), n);
    copied += n;
  }
  glBindBuffer(GL_COPY_READ_BUFFER, 0);
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  if (!ok) {
    REXGPU_WARN("GL RT: GPU resolve write-back skipped, range 0x{:08X}+0x{:X} not resident", pr.extent_start, size);
    ReleaseDownload(pr);
    pr.done = true;
    return;
  }
  pr.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  perf::Add(perf::kLazyDownloads);
  perf::Add(perf::kLazyDownloadBytes, size);
}

void GlRenderTargets::FinishGpuDownload(PendingResolve& pr) {
  perf::Scope perf_dl_scope(perf::kLazyDownloadTicks);
  if (!pr.fence || !pr.pbo) {
    pr.done = true;
    return;
  }
  const void* data = nullptr;
  {
    perf::Scope perf_readback_scope(perf::kReadbackTicks);
    perf::FenceWaitLogged(pr.fence, "resolve write-back");
    glBindBuffer(GL_COPY_WRITE_BUFFER, pr.pbo);
    data = glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, pr.extent_length, GL_MAP_READ_BIT);
  }
  if (data) {
    // The data is already in guest format (tiled, endian swapped): a plain copy.
    std::memcpy(memory_.TranslatePhysical(pr.extent_start), data, pr.extent_length);
    glUnmapBuffer(GL_COPY_WRITE_BUFFER);
  }
  glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
  ReleaseDownload(pr);
  pr.done = true;
}

// ---------------------------------------------------------------------------------------------
// GPU-resident ranges (sorted, merged)
// ---------------------------------------------------------------------------------------------

void GlRenderTargets::ResidentAdd(uint32_t start, uint32_t length) {
  uint64_t s = start, e = uint64_t(start) + length;
  for (size_t i = 0; i < resident_.size();) {
    uint64_t rs = resident_[i].start, re = rs + resident_[i].length;
    if (rs <= e && s <= re) {
      s = std::min(s, rs);
      e = std::max(e, re);
      resident_.erase(resident_.begin() + i);
    } else {
      ++i;
    }
  }
  resident_.push_back({uint32_t(s), uint32_t(e - s)});
  while (resident_.size() > 64) resident_.erase(resident_.begin());
}

void GlRenderTargets::ResidentRemove(uint32_t start, uint32_t length) {
  uint64_t s = start, e = uint64_t(start) + length;
  std::vector<Range> out;
  for (const Range& r : resident_) {
    uint64_t rs = r.start, re = rs + r.length;
    if (re <= s || rs >= e) {
      out.push_back(r);
      continue;
    }
    if (rs < s) out.push_back({r.start, uint32_t(s - rs)});
    if (re > e) out.push_back({uint32_t(e), uint32_t(re - e)});
  }
  resident_ = std::move(out);
}

bool GlRenderTargets::IsGpuResident(uint32_t start, uint32_t length) const {
  uint64_t s = start, e = uint64_t(start) + length;
  for (const Range& r : resident_) {
    if (r.start <= s && uint64_t(r.start) + r.length >= e) {
      // A CPU-path resolve still pending on the range would make the store stale (so would a lazy one:
      // callers MaterializeLazy first).
      for (const PendingResolve& pr : pending_) {
        if ((!pr.gpu || pr.lazy) && uint64_t(pr.extent_start) < e && uint64_t(pr.extent_start) + pr.extent_length > s) return false;
      }
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------------------------
// Pending resolve management
// ---------------------------------------------------------------------------------------------

namespace {
void CountFlush(GlRenderTargets::FlushReason r) {
  using R = GlRenderTargets::FlushReason;
  switch (r) {
    case R::kTex: perf::Add(perf::kFlushTex); break;
    case R::kSm: perf::Add(perf::kFlushSm); break;
    case R::kFront: perf::Add(perf::kFlushFront); break;
    case R::kAge: perf::Add(perf::kFlushAge); break;
    case R::kCap: perf::Add(perf::kFlushCap); break;
    case R::kSync: perf::Add(perf::kFlushSync); break;
    default: perf::Add(perf::kFlushOther); break;
  }
}
}  // namespace

void GlRenderTargets::FlushFront(bool wait, FlushReason reason) {
  if (pending_.empty()) return;
  PendingResolve& pr = pending_.front();
  if (pr.gpu) {
    if (pr.done) {
      ReleasePending(pr);
      pending_.pop_front();
      return;
    }
    if (!wait) return;
    CountFlush(reason);
    if (!pr.dl_started) StartGpuDownload(pr);
    if (!pr.done) FinishGpuDownload(pr);
    ReleasePending(pr);
    pending_.pop_front();
    return;
  }
  if (!wait) {
    GLenum r = glClientWaitSync(pr.fence, 0, 0);
    if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) return;
  } else {
    CountFlush(reason);
  }
  const void* data = nullptr;
  {
    perf::Scope perf_readback_scope(perf::kReadbackTicks);
    perf::FenceWaitLogged(pr.fence, "resolve readback");
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pr.pbo);
    data = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, pr.rw * pr.rh * (pr.src_is_8888 ? 4 : 16), GL_MAP_READ_BIT);
  }
  if (data) {
    ConvertPending(pr, data);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
  }
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  ReleasePending(pr);
  pending_.pop_front();
}

void GlRenderTargets::FlushPending(uint32_t start, uint32_t length, FlushReason reason) {
  if (pending_.empty()) return;
  const uint64_t end = uint64_t(start) + length;
  auto overlaps = [&](const PendingResolve& pr) {
    return uint64_t(pr.extent_start) < end && uint64_t(pr.extent_start) + pr.extent_length > start;
  };
  // CPU-path resolves are written back strictly in order (up to the last overlapping one).
  int last_cpu = -1;
  for (size_t i = 0; i < pending_.size(); ++i) {
    if (!pending_[i].gpu && overlaps(pending_[i])) last_cpu = int(i);
  }
  for (int i = 0; i <= last_cpu; ++i) FlushFront(true, reason);
  // GPU-path resolves: the store always holds the newest data, so each overlapping one is simply downloaded.
  for (PendingResolve& pr : pending_) {
    if (!pr.gpu || pr.done || !overlaps(pr)) continue;
    CountFlush(reason);
    if (!pr.dl_started) StartGpuDownload(pr);
    if (!pr.done) FinishGpuDownload(pr);
  }
}

void GlRenderTargets::FlushAll(FlushReason reason) {
  if (pending_.empty()) return;
  FlushPending(0, 0xFFFFFFFFu, reason);
  while (!pending_.empty()) FlushFront(true, reason);
}

void GlRenderTargets::PollPending() {
  const int64_t max_age = std::max(0, REXCVAR_GET(gl_resolve_max_age));
  while (!pending_.empty()) {
    PendingResolve& pr = pending_.front();
    int64_t age = int64_t(frame_) - int64_t(pr.frame);
    if (pr.gpu) {
      if (pr.done) {
        // Superseded records (no snapshot) go away at once, downloaded ones keep their snapshot for a while.
        if (age >= (pr.snapshot ? max_age : 1)) {
          FlushFront(true, FlushReason::kOther);
          continue;
        }
        break;
      }
      if (!pr.dl_started) {
        if (age < max_age) break;
        StartGpuDownload(pr);
        if (pr.done) continue;
      }
      if (age >= max_age + 3) {
        FlushFront(true, FlushReason::kAge);
        continue;
      }
      GLenum r = glClientWaitSync(pr.fence, 0, 0);
      if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) break;
      FinishGpuDownload(pr);  // keeps the record (and its snapshot) until it ages out
      continue;
    }
    if (age >= max_age) {
      FlushFront(true, FlushReason::kAge);
    } else if (age >= 1) {
      size_t before = pending_.size();
      FlushFront(false);
      if (pending_.size() == before) break;
    } else {
      break;
    }
  }
}

bool GlRenderTargets::FindSnapshot(uint32_t base, uint32_t width, uint32_t height, uint32_t pitch,
                                   uint32_t endian, uint32_t snap_kind, GLuint& texture_out,
                                   bool& swap_rb_out) const {
  for (size_t i = pending_.size(); i-- > 0;) {
    const PendingResolve& pr = pending_[i];
    // The newest resolve overlapping the texture base decides; only an exact match is usable.
    if (pr.snapshot && pr.snap_kind == snap_kind && pr.dest_base == base && pr.rw == width && pr.rh == height &&
        ((pr.dest_pitch + 31) & ~31u) == ((pitch + 31) & ~31u) && uint32_t(pr.endian) == endian) {
      texture_out = pr.snapshot;
      swap_rb_out = pr.swap_rb;
      return true;
    }
    if (pr.extent_start <= base && base < uint64_t(pr.extent_start) + pr.extent_length) return false;
  }
  return false;
}

void GlRenderTargets::LogLookupMiss(const char* who, uint32_t base, uint32_t width, uint32_t height,
                                    uint32_t pitch, uint32_t endian, uint32_t format, uint32_t snap_kind) const {
  const int32_t budget = REXCVAR_GET(gl_resolve_log);
  if (budget <= 0) return;
  static uint32_t counts[2] = {0, 0}, none_counts[2] = {0, 0};
  const int wi = who[0] == 'f' ? 1 : 0;
  // Newest resolve that has this base as its destination, else the newest one covering the base.
  const PendingResolve* match = nullptr;
  for (size_t i = pending_.size(); i-- > 0;) {
    if (pending_[i].dest_base == base) {
      match = &pending_[i];
      break;
    }
  }
  if (!match) {
    for (size_t i = pending_.size(); i-- > 0;) {
      const PendingResolve& pr = pending_[i];
      if (pr.extent_start <= base && base < uint64_t(pr.extent_start) + pr.extent_length) {
        match = &pr;
        break;
      }
    }
  }
  if (!match) {
    // Lookups that no resolve ever targeted are the common case (ordinary textures): only a few are logged.
    uint32_t& nn = none_counts[wi];
    if (++nn > 3 && (nn % 4000) != 0) return;
    uint32_t n = nn;
    REXGPU_INFO("GL lookup miss #{} [{}]: base=0x{:08X} {}x{} pitch={} endian={} fmt={} kind={}: no pending resolve targets this base ({} pending)",
                n, who, base, width, height, pitch, endian, format, snap_kind, pending_.size());
    return;
  }
  uint32_t& cnt = counts[wi];
  ++cnt;
  if (cnt > uint32_t(budget) && (cnt % 1500) != 0) return;
  const uint32_t n = cnt;
  const PendingResolve& pr = *match;
  REXGPU_INFO(
      "GL lookup miss #{} [{}]: want base=0x{:08X} {}x{} pitch={}(al {}) endian={} fmt={} kind={} | resolve dest=0x{:08X} "
      "{}x{} origin=({},{}) pitch={}(al {}) endian={} fmt={} kind={} gpu={} snapshot={} done={} age={} | mismatch:{}{}{}{}{}{}{}",
      n, who, base, width, height, pitch, (pitch + 31) & ~31u, endian, format, snap_kind, pr.dest_base, pr.rw, pr.rh,
      pr.ox, pr.oy, pr.dest_pitch, (pr.dest_pitch + 31) & ~31u, uint32_t(pr.endian), uint32_t(pr.dest_format),
      pr.snap_kind, pr.gpu, pr.snapshot != 0, pr.done, int64_t(frame_) - int64_t(pr.frame),
      pr.dest_base != base ? " base" : "", (pr.rw != width || pr.rh != height) ? " size" : "",
      ((pr.dest_pitch + 31) & ~31u) != ((pitch + 31) & ~31u) ? " pitch" : "",
      uint32_t(pr.endian) != endian ? " endian" : "", pr.snap_kind != snap_kind ? " kind" : "",
      pr.snapshot == 0 ? " nosnapshot" : "", (pr.ox || pr.oy) ? " origin" : "");
}

}  // namespace rex::graphics::gl
