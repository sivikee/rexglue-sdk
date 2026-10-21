/**
 * @file        graphics/gl/render_targets.h
 * @brief       Simplified render target management for the GL backend.
 *
 * Instead of the full EDRAM emulation of the SDK's RenderTargetCache (ownership transfers, dump/load
 * shaders), every EDRAM base tile used as a render target gets its own host texture, keyed by
 * (base tile, format, pitch). MSAA is ignored (1 sample); aliasing of EDRAM regions between different
 * render targets is not tracked.
 *
 * Resolves (copies to guest memory) have two implementations:
 *  - GPU resolve (default, cvar gl_gpu_resolve): a compute shader converts the render target to the guest
 *    format, tiles and endian-swaps it and writes it straight into the shared memory store (the data stays on
 *    the GPU). The guest memory copy is "CPU-stale" and is written back lazily (PBO + fence, memcpy only):
 *    when something reads that guest range on the CPU (FlushPending), or after gl_resolve_max_age swaps, and
 *    not at all if a newer resolve overwrites the range first.
 *  - CPU resolve (fallback): glReadPixels through a PBO and CPU conversion (ConvertPending).
 */
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include <glad/glad.h>

#include <rex/graphics/register_file.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/xenos.h>
#include <rex/memory.h>

namespace rex::graphics::gl {

class GlSharedMemory;

class GlRenderTargets {
 public:
  // Why a pending resolve had to be written back to guest memory early (perf counters).
  enum class FlushReason { kTex, kSm, kFront, kAge, kCap, kSync, kOther };

  struct ColorTarget {
    GLuint texture = 0;
    uint32_t base = 0;
    uint32_t width = 0, height = 0;
    xenos::ColorRenderTargetFormat format = xenos::ColorRenderTargetFormat::k_8_8_8_8;
    GLenum internal_format = GL_RGBA8;
    uint64_t last_used = 0;
  };
  struct DepthTarget {
    GLuint texture = 0;
    uint32_t base = 0;
    uint32_t width = 0, height = 0;
    uint64_t last_used = 0;
  };

  GlRenderTargets(const RegisterFile& register_file, memory::Memory& memory)
      : regs_(register_file), memory_(memory) {}
  ~GlRenderTargets() { Shutdown(); }

  bool Initialize();
  void Shutdown();
  // The shared memory store is the target of GPU resolves; without it only CPU resolves are used.
  void SetSharedMemory(GlSharedMemory* shared_memory) { sm_ = shared_memory; }
  bool gpu_resolve_available() const { return gpu_prog_ != 0; }

  // Binds the framebuffer for the draw. color_mask is the normalized 4-bit-per-RT write mask. Returns
  // false if nothing can be drawn (no targets). width/height_needed are in pixels.
  bool BindForDraw(uint32_t color_mask, bool depth_stencil_used, uint32_t width_needed,
                   uint32_t height_needed);
  uint32_t bound_width() const { return bound_width_; }
  uint32_t bound_height() const { return bound_height_; }
  uint32_t bound_color_mask() const { return bound_color_mask_; }
  bool bound_depth() const { return bound_depth_; }

  // Performs a resolve (copy and/or clear). Returns false on failure; extent_out receives the modified
  // guest physical memory range (length 0 if nothing was written).
  // gpu_written_out is set when the data went to the shared memory store (the caller then must mark the
  // range GPU-written instead of invalidating it).
  bool Resolve(uint32_t& extent_start_out, uint32_t& extent_length_out, bool& gpu_written_out);

  void EndFrame() { ++frame_; }

  // ---- Asynchronous resolves ----
  // Resolves read the render target back through a PBO and are converted to guest memory later. Anything
  // that reads guest memory must call FlushPending first for the range it reads.
  void FlushPending(uint32_t start, uint32_t length, FlushReason reason = FlushReason::kOther);
  void FlushAll(FlushReason reason = FlushReason::kOther);
  // Per-swap housekeeping: converts resolves that are ready / too old.
  void PollPending();
  // A recent, not yet written-back resolve that wrote exactly this 32bpp 2D tiled image (origin 0,0, rw x rh
  // pixels, pitch in pixels, endian). Returns a GL_TEXTURE_2D_ARRAY (1 layer, RGBA8) snapshot of the
  // render target as it was at the resolve. swap_rb: guest byte0 is the snapshot blue channel.
  // snap_kind: 1 = RGBA8, 2 = RGBA16F, 3 = RGB10A2 (see PendingResolve::snap_kind).
  // The swap showed the newest resolve at `base` from its snapshot. With gl_front_writeback = false such a lazy
  // front buffer resolve is dropped when it ages out instead of being read back into guest memory (~40 ms per
  // frame on Mesa nouveau: the two 1024x600 front buffers alternate and age out before being superseded).
  void MarkPresented(uint32_t base);
  bool FindSnapshot(uint32_t base, uint32_t width, uint32_t height, uint32_t pitch, uint32_t endian,
                    uint32_t snap_kind, GLuint& texture_out, bool& swap_rb_out) const;
  // Like FindSnapshot, but the newest resolve at `base` may be LARGER than the texture (resolve rectangles are
  // multiples of 8: Oblivion's HDR luminance chain resolves 256x152 / 8x8 and samples 256x150 / 1x1). The caller
  // copies the top-left width x height part (same tiling: same base and pitch, origin 0).
  bool FindSnapshotContaining(uint32_t base, uint32_t width, uint32_t height, uint32_t pitch, uint32_t endian,
                              uint32_t snap_kind, GLuint& texture_out, bool& swap_rb_out, GLenum& format_out,
                              uint64_t& serial_out) const;
  // Rate-limited diagnostic: why did a consumer's lookup (who = "tex" / "front") not match any resolve?
  void LogLookupMiss(const char* who, uint32_t base, uint32_t width, uint32_t height, uint32_t pitch,
                     uint32_t endian, uint32_t format, uint32_t snap_kind) const;
  // True if the guest range [start, start + length) is entirely held by the shared memory store as the result
  // of GPU resolves (no CPU-path resolve pending on it), so shaders can read it from the store.
  bool IsGpuResident(uint32_t start, uint32_t length) const;
  // Lazy GPU resolves (gl_lazy_resolve): a resolve with an exact GPU snapshot does not run the compute shader
  // at once; the snapshot is the source of truth until something reads the guest range through the shared
  // memory store (vertex / index fetch, store blit) or guest memory (FlushPending), or it ages out. Superseded
  // lazy resolves are never written at all. Call before reading [start, start + length) from the store.
  void MaterializeLazy(uint32_t start, uint32_t length) {
    if (lazy_count_) MaterializeLazySlow(start, length);
  }
  void ClearGpuResident() { resident_.clear(); }
  bool DebugReadColor(uint32_t base, std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h);

  struct PendingResolve {
    uint64_t frame = 0;
    uint32_t dest_base = 0, extent_start = 0, extent_length = 0;
    uint32_t ox = 0, oy = 0, rw = 0, rh = 0, dest_pitch = 0;
    xenos::TextureFormat dest_format = xenos::TextureFormat::k_8_8_8_8;
    uint32_t bpp = 4, bpp_log2 = 2;
    xenos::Endian128 endian = xenos::Endian128::kNone;
    bool swap_rb = false, src_is_8888 = false;
    GLuint pbo = 0;
    uint32_t pbo_size = 0;
    GLsync fence = nullptr;
    GLuint snapshot = 0;
    uint32_t snapshot_w = 0, snapshot_h = 0;
    GLenum snapshot_format = 0;
    uint32_t snap_kind = 0;  // 0 none, 1 RGBA8, 2 RGBA16F, 3 RGB10A2, 4 R32F
    bool gpu = false;        // data was resolved into the shared memory store by the compute shader
    bool done = false;       // nothing left to write back (superseded by a newer resolve)
    bool dl_started = false; // gpu: store -> PBO copy issued, fence pending
    int gpu_kind = -1;       // gpu: dest layout (see kGpuKind*)
    bool lazy = false;       // gpu: compute write into the store deferred (data only in the snapshot)
    uint64_t presented_frame = UINT64_MAX;  // last swap that showed it straight from its snapshot (MarkPresented)
    uint64_t serial = 0;     // resolve number (snapshot contents identity for consumers' derived copies)
  };
  struct Snapshot {
    GLuint texture = 0;
    uint32_t w = 0, h = 0;
    GLenum format = 0;
  };
  struct Range {
    uint32_t start, length;
  };
  void ConvertPending(PendingResolve& pr, const void* pixels);
  void FlushFront(bool wait, FlushReason reason = FlushReason::kOther);
  void ReleasePending(PendingResolve& pr);
  void ReleaseDownload(PendingResolve& pr);
  void ReleaseSnapshot(PendingResolve& pr);
  void StartGpuDownload(PendingResolve& pr);
  void FinishGpuDownload(PendingResolve& pr);

 private:
  struct FbKey {
    GLuint color[4];
    GLuint depth;
    bool operator<(const FbKey& o) const {
      return std::tie(color[0], color[1], color[2], color[3], depth) <
             std::tie(o.color[0], o.color[1], o.color[2], o.color[3], o.depth);
    }
  };

  ColorTarget* GetColorTarget(uint32_t base, xenos::ColorRenderTargetFormat format, uint32_t width,
                              uint32_t height, bool create);
  DepthTarget* GetDepthTarget(uint32_t base, uint32_t width, uint32_t height, bool create);
  static GLenum GetInternalFormat(xenos::ColorRenderTargetFormat format);
  bool CreateGpuResolveProgram();
  bool GpuResolve(ColorTarget& rt, PendingResolve& pr, uint32_t x0, uint32_t y0);
  // Runs the resolve compute shader for pr, reading texture (2D render target at x0,y0 or the 2D array snapshot).
  void GpuResolveDispatch(PendingResolve& pr, GLuint texture, bool from_snapshot, uint32_t x0, uint32_t y0);
  void Materialize(PendingResolve& pr, const char* why);
  // gl_gpu_resolve_compute = false: a lazy GPU-snapshot resolve that must reach memory becomes a CPU-path record
  // (asynchronous glReadPixels of its snapshot into a PBO, CPU conversion when flushed); its store pages are
  // invalidated so they are re-read from guest memory.
  void ConvertLazyToCpu(PendingResolve& pr, const char* why);
  void MaterializeLazySlow(uint32_t start, uint32_t length);
  // Removes superseded records (nothing to write, no snapshot) anywhere in the queue.
  void CompactSuperseded();
  void ResidentAdd(uint32_t start, uint32_t length);
  void ResidentRemove(uint32_t start, uint32_t length);
  GLuint GetFramebuffer(const FbKey& key);
  void InvalidateFramebuffersUsing(GLuint texture);

  const RegisterFile& regs_;
  memory::Memory& memory_;
  std::map<std::pair<uint32_t, uint32_t>, ColorTarget> color_targets_;  // key: (base, format)
  std::map<uint32_t, DepthTarget> depth_targets_;                       // key: base
  std::map<FbKey, GLuint> framebuffers_;
  GLuint read_fbo_ = 0;
  uint32_t bound_width_ = 0, bound_height_ = 0, bound_color_mask_ = 0;
  bool bound_depth_ = false;
  uint64_t frame_ = 0;
  uint32_t resolve_count_ = 0;
  std::deque<PendingResolve> pending_;
  std::vector<std::pair<GLuint, uint32_t>> free_pbos_;
  std::vector<Snapshot> free_snapshots_;
  GlSharedMemory* sm_ = nullptr;
  GLuint gpu_prog_ = 0;
  GLuint gpu_prog_array_ = 0;  // same shader reading a GL_TEXTURE_2D_ARRAY snapshot (lazy materialization)
  GLint ua_src_ = -1, ua_dst_ = -1, ua_base_ = -1, ua_fmt_ = -1;
  uint32_t lazy_count_ = 0;   // pending resolves with lazy == true
  int gpu_checks_left_ = 8;
  GLint u_src_ = -1, u_dst_ = -1, u_base_ = -1, u_fmt_ = -1;
  std::vector<uint64_t> resolve_sigs_;  // diagnostics: resolve configurations already logged
  std::vector<Range> resident_;  // guest ranges whose current contents live in the store (GPU resolves)
};

}  // namespace rex::graphics::gl
