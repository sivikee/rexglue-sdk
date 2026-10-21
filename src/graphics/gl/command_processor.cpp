#include "command_processor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>

#include <switch.h>
#include <rex/input/switch/switch_input_driver.h>
#include <xxhash.h>

#include <rex/cvar.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/pipeline/shader/spirv.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/memory.h>

#include "gl_ext.h"
#include "perf.h"
#include "pipeline_cache.h"
#include "primitive_processor.h"
#include "render_targets.h"
#include "gl_cvars.h"
#include "shared_memory.h"
#include "stream_ring.h"
#include "tiling.h"
#include "texture_cache.h"

REXCVAR_DEFINE_STRING(gl_screenshot_dir, "", "GPU",
                      "GL backend: directory for presented-frame screenshots (.ppm); empty disables");
REXCVAR_DEFINE_INT32(gl_screenshot_swaps, 0, "GPU",
                     "GL backend: write a screenshot every N swaps (0 = only gl_screenshot_first)");
REXCVAR_DEFINE_INT32(gl_screenshot_from, 0, "GPU",
                     "GL backend: from this swap index on, screenshot every gl_screenshot_from_every swaps (0 = off)");
REXCVAR_DEFINE_INT32(gl_screenshot_from_every, 5, "GPU", "GL backend: interval used with gl_screenshot_from");
REXCVAR_DEFINE_INT32(gl_screenshot_first, 1, "GPU",
                     "GL backend: also screenshot the swap with this index (1-based, 0 = none)");
REXCVAR_DEFINE_BOOL(switch_fps_overlay, true, "GPU", "Switch: draw an FPS / frame-time overlay in the top-left corner");
REXCVAR_DEFINE_BOOL(gl_precompile_screen, true, "GPU", "GL backend: at start-up, show a \"recompiling shaders\" screen and hold the game until every program of the program cache is loaded or compiled");
REXCVAR_DEFINE_INT32(gl_precompile_max_s, 0, "GPU", "GL backend: stop waiting on the start-up pre-compilation after this many seconds (0 = wait until done)");
REXCVAR_DEFINE_BOOL(gl_stream_uploads, true, "GPU", "GL backend: upload constants / shared memory / textures through unsynchronised streaming rings (avoids nouveau pipeline stalls); false = old glBufferSubData paths");
REXCVAR_DEFINE_BOOL(gl_sync_resolve, false, "GPU", "GL backend: resolve (EDRAM copy) readback is synchronous like the original implementation (debugging)");
REXCVAR_DEFINE_BOOL(gl_resolve_gpu_copy, true, "GPU", "GL backend: textures / front buffer matching a pending resolve use a GPU-side copy of the render target instead of guest memory");
REXCVAR_DEFINE_INT32(gl_resolve_max_age, 3, "GPU", "GL backend: a pending async resolve is written to guest memory after at most this many swaps");
REXCVAR_DEFINE_BOOL(gl_state_cache, true, "GPU", "GL backend: skip redundant raster state GL calls");
REXCVAR_DEFINE_BOOL(gl_gpu_resolve, true, "GPU", "GL backend: resolves (EDRAM copies) are done by a compute shader into the shared memory store; guest memory is written back lazily (false = CPU readback + conversion)");
REXCVAR_DEFINE_BOOL(gl_gpu_front, true, "GPU", "GL backend: the front buffer is detiled / endian swapped by a fragment shader reading the shared memory store (false = other paths)");
REXCVAR_DEFINE_BOOL(gl_cpu_frontbuffer, false, "GPU", "GL backend: force the old CPU detile + upload of the front buffer (debugging)");
REXCVAR_DEFINE_INT32(gl_resolve_log, 12, "GPU", "GL backend: log budget for resolve configurations and failed GPU-copy lookups (0 = off)")
;
REXCVAR_DEFINE_BOOL(gl_gpu_timers, true, "GPU", "GL backend: GPU timestamp queries + frame fences for the perf report (GPU-side present / resolve time, GPU lag)");
REXCVAR_DEFINE_BOOL(gl_lazy_resolve, true, "GPU", "GL backend: resolves with an exact GPU snapshot defer the compute write into the shared memory store until the range is read (store / guest memory) or ages out; superseded ones are never written");
REXCVAR_DEFINE_BOOL(gl_present_snapshot, true, "GPU", "GL backend: present the front buffer straight from the resolve snapshot when one matches (plain textured blit) before trying the store detile blit / CPU path");
REXCVAR_DEFINE_DOUBLE(gl_gpu_timestamp_scale, 0.0, "GPU", "GL backend: ns per raw GL_TIMESTAMP unit (0 = auto: 625/384 on Switch, whose GPU timer ticks at 614.4 MHz and Mesa returns raw ticks)");
REXCVAR_DEFINE_INT32(gl_swap_interval, 1, "GPU", "GL backend: eglSwapInterval (1 = wait for vsync; 0 = present without waiting for vsync)");
REXCVAR_DEFINE_BOOL(gl_draw, true, "GPU", "GL backend: execute draws (false = swaps/resolves only)");
REXCVAR_DEFINE_BOOL(gl_flip_front_face, false, "GPU", "GL backend: invert the front face winding");
REXCVAR_DEFINE_INT32(gl_log_draws, 0, "GPU", "GL backend: log the first N draws in detail");
REXCVAR_DEFINE_INT32(gl_log_draws_from, 0, "GPU", "GL backend: also log 12 draws starting at this draw index (0 = off)");
REXCVAR_DEFINE_INT32(gl_debug_raster, 0, "GPU", "GL debug bitmask: 1=no scissor 2=no cull 4=force colour write/no blend 8=no depth/stencil 16=viewport=full RT");
REXCVAR_DEFINE_INT32(gl_max_viewport, 16384, "GPU", "GL backend: max viewport dimension used when computing the host viewport");
REXCVAR_DEFINE_INT32(gl_dump_rt_swaps, 0, "GPU", "GL backend: dump colour RT 0 to rt_<swap>.ppm every N swaps");

namespace rex::graphics {
extern std::atomic<uint32_t> g_switch_vdswap_count;  // base command_processor.cpp, bumped by VdSwap
// Set once the GPU thread is set up and the start-up shader pre-compilation is over; the Switch harness starts the
// guest main thread only after that, so the intro does not play while programs compile.
std::atomic<bool> g_switch_shaders_ready{false};
}  // namespace rex::graphics

namespace rex::graphics::gl {

namespace {
const char* kBlitVs = R"(#version 430 core
out vec2 uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
const char* kBlitFs = R"(#version 430 core
in vec2 uv;
layout(binding = 0) uniform sampler2D tex;
out vec4 o;
void main() { o = vec4(texture(tex, uv).rgb, 1.0); }
)";

GLuint Compile(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, sizeof(log), nullptr, log);
    REXGPU_ERROR("GL: shader compile failed: {}", log);
  }
  return s;
}

void WritePpm(const std::string& path, int w, int h) {
  std::vector<uint8_t> rgba(size_t(w) * h * 4);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    REXGPU_WARN("GL: cannot write screenshot {}", path);
    return;
  }
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  std::vector<uint8_t> row(size_t(w) * 3);
  for (int y = h - 1; y >= 0; --y) {
    const uint8_t* src = &rgba[size_t(y) * w * 4];
    for (int x = 0; x < w; ++x) {
      row[x * 3 + 0] = src[x * 4 + 0];
      row[x * 3 + 1] = src[x * 4 + 1];
      row[x * 3 + 2] = src[x * 4 + 2];
    }
    fwrite(row.data(), 1, row.size(), f);
  }
  fclose(f);
  REXGPU_INFO("GL: screenshot {} ({}x{})", path, w, h);
}

GLenum CompareFunc(xenos::CompareFunction f) {
  static const GLenum map[8] = {GL_NEVER, GL_LESS,    GL_EQUAL,  GL_LEQUAL,
                                GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS};
  return map[uint32_t(f) & 7];
}

GLenum StencilOpGl(xenos::StencilOp op) {
  static const GLenum map[8] = {GL_KEEP, GL_ZERO, GL_REPLACE, GL_INCR, GL_DECR, GL_INVERT, GL_INCR_WRAP, GL_DECR_WRAP};
  return map[uint32_t(op) & 7];
}

GLenum BlendFactorGl(xenos::BlendFactor f) {
  switch (f) {
    case xenos::BlendFactor::kZero: return GL_ZERO;
    case xenos::BlendFactor::kOne: return GL_ONE;
    case xenos::BlendFactor::kSrcColor: return GL_SRC_COLOR;
    case xenos::BlendFactor::kOneMinusSrcColor: return GL_ONE_MINUS_SRC_COLOR;
    case xenos::BlendFactor::kSrcAlpha: return GL_SRC_ALPHA;
    case xenos::BlendFactor::kOneMinusSrcAlpha: return GL_ONE_MINUS_SRC_ALPHA;
    case xenos::BlendFactor::kDstColor: return GL_DST_COLOR;
    case xenos::BlendFactor::kOneMinusDstColor: return GL_ONE_MINUS_DST_COLOR;
    case xenos::BlendFactor::kDstAlpha: return GL_DST_ALPHA;
    case xenos::BlendFactor::kOneMinusDstAlpha: return GL_ONE_MINUS_DST_ALPHA;
    case xenos::BlendFactor::kConstantColor: return GL_CONSTANT_COLOR;
    case xenos::BlendFactor::kOneMinusConstantColor: return GL_ONE_MINUS_CONSTANT_COLOR;
    case xenos::BlendFactor::kConstantAlpha: return GL_CONSTANT_ALPHA;
    case xenos::BlendFactor::kOneMinusConstantAlpha: return GL_ONE_MINUS_CONSTANT_ALPHA;
    case xenos::BlendFactor::kSrcAlphaSaturate: return GL_SRC_ALPHA_SATURATE;
    default: return GL_ZERO;
  }
}

GLenum BlendOpGl(xenos::BlendOp op) {
  switch (op) {
    case xenos::BlendOp::kAdd: return GL_FUNC_ADD;
    case xenos::BlendOp::kSubtract: return GL_FUNC_SUBTRACT;
    case xenos::BlendOp::kMin: return GL_MIN;
    case xenos::BlendOp::kMax: return GL_MAX;
    case xenos::BlendOp::kRevSubtract: return GL_FUNC_REVERSE_SUBTRACT;
    default: return GL_FUNC_ADD;
  }
}

// ---- tiny 5x7 bitmap font for the FPS overlay (digits, '.', '-' and the letters the overlay prints) ----
const uint8_t* OverlayGlyph(char ch) {
  static const struct { char c; uint8_t r[7]; } kGlyphs[] = {
      {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
      {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {'3', {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E}},
      {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
      {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
      {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
      {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}}, {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}},
      {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}}, {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
      {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}}, {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
      {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}}, {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
      {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}}, {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}},
      {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}}, {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11}},
      {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}}, {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}},
      {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}}, {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
      {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}}, {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}},
      {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}}, {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}},
      {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}}, {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
      {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}}, {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}},
      {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}}, {'/', {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}},
      {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}}, {'%', {0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13}},
  };
  for (const auto& g : kGlyphs) {
    if (g.c == ch) return g.r;
  }
  return nullptr;
}

// Redundant-state filter for the raster state set per draw. Anything outside ApplyRasterState that changes
// this state (resolves, swap) must call Invalidate().
struct RasterStateCache {
  bool enabled = true;
  static constexpr int kCaps = 10;
  GLenum cap_ids[kCaps] = {GL_SCISSOR_TEST, GL_DEPTH_CLAMP, GL_CULL_FACE, GL_POLYGON_OFFSET_FILL,
                           GL_POLYGON_OFFSET_LINE, GL_POLYGON_OFFSET_POINT, GL_DEPTH_TEST, GL_STENCIL_TEST,
                           GL_BLEND, 0};
  int8_t cap_state[kCaps];
  int8_t blend_state[4];
  // Values.
  bool vp_valid = false, sc_valid = false, dr_valid = false, po_valid = false, bc_valid = false;
  GLint vp[4], sc[4];
  double dr[2];
  float po[2], bc[4];
  GLenum cull = 0, front = 0, polymode = 0, depth_func = 0;
  int8_t depth_mask = -1;
  struct Stencil {
    bool valid = false;
    GLenum func, fail, zfail, zpass;
    GLint ref;
    GLuint mask, wmask;
  } stencil[2];
  struct Rt {
    int8_t mask = -1;
    bool blend_valid = false;
    GLenum fs, fd, as, ad, ec, ea;
  } rt[4];

  RasterStateCache() { Invalidate(); }
  void Invalidate() {
    for (auto& c : cap_state) c = -1;
    for (auto& c : blend_state) c = -1;
    vp_valid = sc_valid = dr_valid = po_valid = bc_valid = false;
    cull = front = polymode = depth_func = 0;
    depth_mask = -1;
    for (auto& s : stencil) s.valid = false;
    for (auto& r : rt) r = Rt();
  }
  void Cap(GLenum cap, bool on) {
    if (enabled) {
      for (int i = 0; i < kCaps - 2; ++i) {
        if (cap_ids[i] == cap) {
          if (cap_state[i] == int8_t(on)) return;
          cap_state[i] = int8_t(on);
          break;
        }
      }
    }
    if (on) glEnable(cap); else glDisable(cap);
  }
  void Viewport(GLint x, GLint y, GLsizei w, GLsizei h) {
    if (enabled && vp_valid && vp[0] == x && vp[1] == y && vp[2] == w && vp[3] == h) return;
    vp[0] = x; vp[1] = y; vp[2] = w; vp[3] = h; vp_valid = true;
    glViewport(x, y, w, h);
  }
  void Scissor(GLint x, GLint y, GLsizei w, GLsizei h) {
    if (enabled && sc_valid && sc[0] == x && sc[1] == y && sc[2] == w && sc[3] == h) return;
    sc[0] = x; sc[1] = y; sc[2] = w; sc[3] = h; sc_valid = true;
    glScissor(x, y, w, h);
  }
  void DepthRange(double n, double f) {
    if (enabled && dr_valid && dr[0] == n && dr[1] == f) return;
    dr[0] = n; dr[1] = f; dr_valid = true;
    glDepthRange(n, f);
  }
  void CullFace(GLenum m) { if (enabled && cull == m) return; cull = m; glCullFace(m); }
  void FrontFace(GLenum m) { if (enabled && front == m) return; front = m; glFrontFace(m); }
  void PolygonMode(GLenum m) { if (enabled && polymode == m) return; polymode = m; glPolygonMode(GL_FRONT_AND_BACK, m); }
  void PolygonOffset(float slope, float constant) {
    if (enabled && po_valid && po[0] == slope && po[1] == constant) return;
    po[0] = slope; po[1] = constant; po_valid = true;
    glPolygonOffset(slope, constant);
  }
  void DepthFunc(GLenum f) { if (enabled && depth_func == f) return; depth_func = f; glDepthFunc(f); }
  void DepthMask(bool m) { if (enabled && depth_mask == int8_t(m)) return; depth_mask = int8_t(m); glDepthMask(m ? GL_TRUE : GL_FALSE); }
  void StencilSide(int side, GLenum func, GLint ref, GLuint mask, GLenum fail, GLenum zfail, GLenum zpass, GLuint wmask) {
    Stencil& s = stencil[side];
    if (enabled && s.valid && s.func == func && s.ref == ref && s.mask == mask && s.fail == fail && s.zfail == zfail &&
        s.zpass == zpass && s.wmask == wmask) {
      return;
    }
    s = {true, func, fail, zfail, zpass, ref, mask, wmask};
    GLenum face = side ? GL_BACK : GL_FRONT;
    glStencilFuncSeparate(face, func, ref, mask);
    glStencilOpSeparate(face, fail, zfail, zpass);
    glStencilMaskSeparate(face, wmask);
  }
  void BlendColor(const float* c) {
    if (enabled && bc_valid && !std::memcmp(bc, c, sizeof(bc))) return;
    std::memcpy(bc, c, sizeof(bc)); bc_valid = true;
    glBlendColor(c[0], c[1], c[2], c[3]);
  }
  void ColorMask(uint32_t i, uint32_t mask) {
    if (enabled && rt[i].mask == int8_t(mask)) return;
    rt[i].mask = int8_t(mask);
    glColorMaski(i, mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, (mask >> 3) & 1);
  }
  void BlendEnable(uint32_t i, bool on) {
    if (enabled && blend_state[i] == int8_t(on)) return;
    blend_state[i] = int8_t(on);
    if (on) glEnablei(GL_BLEND, i); else glDisablei(GL_BLEND, i);
  }
  void BlendFunc(uint32_t i, GLenum fs, GLenum fd, GLenum as, GLenum ad, GLenum ec, GLenum ea) {
    Rt& r = rt[i];
    if (enabled && r.blend_valid && r.fs == fs && r.fd == fd && r.as == as && r.ad == ad && r.ec == ec && r.ea == ea) return;
    r.blend_valid = true; r.fs = fs; r.fd = fd; r.as = as; r.ad = ad; r.ec = ec; r.ea = ea;
    glBlendFuncSeparatei(i, fs, fd, as, ad);
    glBlendEquationSeparatei(i, ec, ea);
  }
};

}  // namespace

struct GlCommandProcessor::Impl {
  // Present
  GLuint blit_program = 0;
  GLuint empty_vao = 0;
  GLuint front_tex = 0;
  GLuint dbg_program = 0;
  // Two overlay textures, alternated on every text change (once per second): the one rewritten was last sampled a
  // second ago, so the upload never waits for the GPU to finish with a texture still in use.
  GLuint overlay_tex[2] = {0, 0};
  int overlay_cur = 0;
  uint32_t overlay_w = 0, overlay_h = 0;
  int gpu_frame_begin = -1;  // GPU timestamp handle issued after the previous present
  std::string overlay_text_cached;
  uint32_t front_w = 0, front_h = 0;
  std::vector<uint32_t> front_pixels;

  // Draw
  std::unique_ptr<GlSharedMemory> shared_memory;
  std::unique_ptr<GlPrimitiveProcessor> primitive_processor;
  std::unique_ptr<GlPipelineCache> pipeline_cache;
  std::unique_ptr<GlRenderTargets> render_targets;
  std::unique_ptr<GlTextureCache> texture_cache;
  GLuint draw_vao = 0;
  GLuint ubo_ring = 0;
  static constexpr uint32_t kUboRingSize = 8u << 20;
  uint32_t ubo_ring_offset = 0;
  std::vector<float> floats_vs, floats_ps;
  StreamRing const_ring;  // uniform streaming (gl_stream_uploads)
  RasterStateCache raster_cache;
  StreamRing tex_ring;    // pixel-unpack streaming (textures, front buffer)
  bool stream = false;
  uint32_t ubo_align = 256;
  GLuint blit_array_program = 0;
  GLint blit_array_sel_loc = -1;
  GLuint blit_store_program = 0;
  GLint blit_store_a_loc = -1, blit_store_b_loc = -1, blit_store_sel_loc = -1;
  GlTextureCache::FetchInfo fetch_info[32];
  std::unordered_map<uint64_t, uint64_t> vertex_buffer_hashes;
  uint32_t texture_bound_mask = 0;
};

GlCommandProcessor::GlCommandProcessor(GraphicsSystem* graphics_system,
                                       system::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state), impl_(std::make_unique<Impl>()) {
  std::memset(&system_constants_, 0, sizeof(system_constants_));
}

GlCommandProcessor::~GlCommandProcessor() = default;

bool GlCommandProcessor::SetupContext() {
  struct ReadyOnExit {
    ~ReadyOnExit() { g_switch_shaders_ready = true; }  // also on failure: the guest must not wait forever
  } ready_on_exit;
  if (!context_.Initialize()) {
    return false;
  }
  glGenVertexArrays(1, &impl_->empty_vao);
  GLuint vs = Compile(GL_VERTEX_SHADER, kBlitVs);
  GLuint fs = Compile(GL_FRAGMENT_SHADER, kBlitFs);
  impl_->blit_program = glCreateProgram();
  glAttachShader(impl_->blit_program, vs);
  glAttachShader(impl_->blit_program, fs);
  glLinkProgram(impl_->blit_program);
  glDeleteShader(vs);
  glDeleteShader(fs);
  {
    const char* fs_arr = R"(#version 430 core
in vec2 uv;
layout(binding = 0) uniform sampler2DArray tex;
uniform ivec3 sel;
out vec4 o;
float pick(vec4 c, int s) { return s < 4 ? c[s] : float(s == 5); }
void main() { vec4 c = texture(tex, vec3(uv, 0.0)); o = vec4(pick(c, sel.x), pick(c, sel.y), pick(c, sel.z), 1.0); }
)";
    GLuint avs = Compile(GL_VERTEX_SHADER, kBlitVs);
    GLuint afs = Compile(GL_FRAGMENT_SHADER, fs_arr);
    impl_->blit_array_program = glCreateProgram();
    glAttachShader(impl_->blit_array_program, avs);
    glAttachShader(impl_->blit_array_program, afs);
    glLinkProgram(impl_->blit_array_program);
    glDeleteShader(avs);
    glDeleteShader(afs);
    impl_->blit_array_sel_loc = glGetUniformLocation(impl_->blit_array_program, "sel");
  }
  {
    // Front buffer straight from the shared memory store: tiled 32bpp address, endian swap, format, swizzle and a
    // bilinear filter, all in the fragment shader (no CPU detile / upload).
    const char* fs_store = R"(#version 430 core
in vec2 uv;
layout(binding = 30) uniform usamplerBuffer xe_sm_store;
layout(binding = 31) uniform usamplerBuffer xe_sm_pt;
uniform uvec4 u_a;   // base address, pitch aligned to 32, width, height
uniform ivec4 u_b;   // endian (0..3), format (0 = 8888, 1 = 2_10_10_10)
uniform ivec3 u_sel; // fetch constant swizzle (0..3 = component, 4 = 0, 5 = 1)
out vec4 o;

uint tiled(uint x, uint y, uint pitch) {
  int macro = int((x >> 5) + (y >> 5) * (pitch >> 5)) << 9;
  int micro = int((x & 7u) + ((y & 0xEu) << 2)) << 2;
  int offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + int((y & 1u) << 4);
  return uint(((offset & ~0x1FF) << 3) + int((y & 16u) << 7) + ((offset & 0x1C0) << 2) +
              int(((((y & 8u) >> 2) + (x >> 3)) & 3u) << 6) + (offset & 0x3F));
}

vec4 fetch_texel(ivec2 p) {
  uint addr = u_a.x + tiled(uint(p.x), uint(p.y), u_a.y);
  uint slot = texelFetch(xe_sm_pt, int(addr >> 20)).x;
  uint v = texelFetch(xe_sm_store, int(slot + ((addr & 0xFFFFFu) >> 2))).x;
  int e = u_b.x;
  if (e == 1) v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
  else if (e == 2) v = (v << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
  else if (e == 3) v = (v << 16) | (v >> 16);
  if (u_b.y == 1) {
    return vec4(float(v & 1023u), float((v >> 10) & 1023u), float((v >> 20) & 1023u), float(v >> 30)) *
           vec4(1.0 / 1023.0, 1.0 / 1023.0, 1.0 / 1023.0, 1.0 / 3.0);
  }
  return vec4(float(v & 255u), float((v >> 8) & 255u), float((v >> 16) & 255u), float(v >> 24)) * (1.0 / 255.0);
}

float pick(vec4 c, int s) { return s < 4 ? c[s] : float(s == 5); }

void main() {
  vec2 p = uv * vec2(u_a.zw) - 0.5;
  ivec2 i0 = ivec2(floor(p));
  vec2 f = p - vec2(i0);
  ivec2 hi = ivec2(u_a.zw) - 1;
  vec4 c00 = fetch_texel(clamp(i0, ivec2(0), hi));
  vec4 c10 = fetch_texel(clamp(i0 + ivec2(1, 0), ivec2(0), hi));
  vec4 c01 = fetch_texel(clamp(i0 + ivec2(0, 1), ivec2(0), hi));
  vec4 c11 = fetch_texel(clamp(i0 + ivec2(1, 1), ivec2(0), hi));
  vec4 c = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
  o = vec4(pick(c, u_sel.x), pick(c, u_sel.y), pick(c, u_sel.z), 1.0);
}
)";
    GLuint svs = Compile(GL_VERTEX_SHADER, kBlitVs);
    GLuint sfs = Compile(GL_FRAGMENT_SHADER, fs_store);
    impl_->blit_store_program = glCreateProgram();
    glAttachShader(impl_->blit_store_program, svs);
    glAttachShader(impl_->blit_store_program, sfs);
    glLinkProgram(impl_->blit_store_program);
    glDeleteShader(svs);
    glDeleteShader(sfs);
    GLint linked = 0;
    glGetProgramiv(impl_->blit_store_program, GL_LINK_STATUS, &linked);
    if (!linked) {
      REXGPU_ERROR("GL: front buffer store blit program failed to link, gl_gpu_front disabled");
      glDeleteProgram(impl_->blit_store_program);
      impl_->blit_store_program = 0;
    } else {
      impl_->blit_store_a_loc = glGetUniformLocation(impl_->blit_store_program, "u_a");
      impl_->blit_store_b_loc = glGetUniformLocation(impl_->blit_store_program, "u_b");
      impl_->blit_store_sel_loc = glGetUniformLocation(impl_->blit_store_program, "u_sel");
    }
  }
  glGenTextures(1, &impl_->front_tex);
  {
    const char* vs_src = "#version 430 core\nvoid main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";
    const char* fs_src = "#version 430 core\nlayout(location=0) out vec4 o;\nvoid main(){ o = vec4(1.0, 0.0, 1.0, 1.0); }\n";
    GLuint dvs = Compile(GL_VERTEX_SHADER, vs_src), dfs = Compile(GL_FRAGMENT_SHADER, fs_src);
    impl_->dbg_program = glCreateProgram();
    glAttachShader(impl_->dbg_program, dvs);
    glAttachShader(impl_->dbg_program, dfs);
    glLinkProgram(impl_->dbg_program);
  }

  // Draw path.
  if (glClipControl_) {
    glClipControl_(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
  } else {
    REXGPU_ERROR("GL: ARB_clip_control missing, depth will be wrong");
  }
  glGenVertexArrays(1, &impl_->draw_vao);
  glGenBuffers(1, &impl_->ubo_ring);
  glBindBuffer(GL_UNIFORM_BUFFER, impl_->ubo_ring);
  glBufferData(GL_UNIFORM_BUFFER, Impl::kUboRingSize, nullptr, GL_STREAM_DRAW);
{    GLint ua = 256;    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &ua);    impl_->ubo_align = uint32_t(std::max(ua, 16));    impl_->stream = REXCVAR_GET(gl_stream_uploads);    if (impl_->stream) {      bool ok = impl_->const_ring.Initialize(16u << 20, 16, GL_UNIFORM_BUFFER, true) &&                impl_->tex_ring.Initialize(48u << 20, 12, GL_PIXEL_UNPACK_BUFFER, true);      if (!ok) {        REXGPU_WARN("GL: streaming rings unavailable, using synchronous uploads");        impl_->const_ring.Shutdown();        impl_->tex_ring.Shutdown();        impl_->stream = false;      }    }  }

  impl_->shared_memory = std::make_unique<GlSharedMemory>(*memory_);
  impl_->primitive_processor =
      std::make_unique<GlPrimitiveProcessor>(*register_file_, *memory_, *impl_->shared_memory);
  impl_->pipeline_cache = std::make_unique<GlPipelineCache>(*register_file_);
  impl_->render_targets = std::make_unique<GlRenderTargets>(*register_file_, *memory_);
  impl_->texture_cache = std::make_unique<GlTextureCache>(*register_file_, *memory_);
  impl_->texture_cache->SetRenderTargets(impl_->render_targets.get());
  impl_->texture_cache->SetStreamRing(impl_->stream ? &impl_->tex_ring : nullptr);
  impl_->render_targets->SetSharedMemory(impl_->shared_memory.get());
  impl_->shared_memory->SetPreReadHook([this](uint32_t start, uint32_t length) {
    if (impl_->render_targets) {
      impl_->render_targets->FlushPending(start, length, GlRenderTargets::FlushReason::kSm);
    }
  });
  impl_->shared_memory->SetPreStoreResetHook([this]() {
    // The store is about to be discarded: GPU-only data must reach guest memory first.
    if (impl_->render_targets) {
      impl_->render_targets->FlushAll(GlRenderTargets::FlushReason::kOther);
      impl_->render_targets->ClearGpuResident();
    }
  });
  if (!impl_->shared_memory->Initialize() || !impl_->primitive_processor->Initialize() ||
      !impl_->pipeline_cache->Initialize(&context_) || !impl_->render_targets->Initialize() ||
      !impl_->texture_cache->Initialize()) {
    REXGPU_ERROR("GL: failed to initialise the draw subsystems");
    return false;
  }

  // Fixed GL buffers (Mesa nouveau allocates GPU memory from the newlib heap on the Switch).
  perf::MemAdd(perf::kMemBuffers, int64_t(Impl::kUboRingSize) + (impl_->stream ? int64_t((16u + 48u) << 20) : 0));
  impl_->pipeline_cache->WarmUpFromDisk();
  ShaderPrecompileScreen();
  {
    const int interval = std::clamp(REXCVAR_GET(gl_swap_interval), 0, 4);
    if (interval != 1) context_.SetSwapInterval(interval);
    REXGPU_INFO("GL: swap interval {}", interval);
  }
  perf::GpuTimersInit();

  // Show something immediately so the window is not stale.
  glClearColor(0.f, 0.f, 0.25f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);
  context_.SwapBuffers();
  REXGPU_INFO("GL: command processor ready");
  return true;
}

void GlCommandProcessor::ShutdownContext() {
  impl_->const_ring.Shutdown();
  impl_->tex_ring.Shutdown();
  impl_->texture_cache.reset();
  impl_->render_targets.reset();
  impl_->pipeline_cache.reset();
  impl_->primitive_processor.reset();
  impl_->shared_memory.reset();
  context_.Shutdown();
}

Shader* GlCommandProcessor::LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                                       const uint32_t* host_address, uint32_t dword_count) {
  if (shader_type == xenos::ShaderType::kVertex && dword_count == 24 && host_address[9] == 0x2 + 0x0010F823 && false) {}
  {
    static bool scanned = false;
    if (!scanned && shader_type == xenos::ShaderType::kVertex && dword_count == 24 && host_address[0] == 0 && host_address[9] == 0x0010F825u) {
      scanned = true;
      uint64_t total = 0; int hits = 0;
      uint32_t addr = 0x40000000u;
      while (addr < 0x90000000u && hits < 8) {
        auto* heap = memory_->LookupHeap(addr);
        memory::HeapAllocationInfo info;
        if (!heap || !heap->QueryRegionInfo(addr, &info) || !info.region_size) { addr += 0x10000; continue; }
        uint32_t rsz = info.region_size;
        if ((info.state & memory::kMemoryAllocationCommit) && info.protect) {
          const uint32_t* base = reinterpret_cast<const uint32_t*>(memory_->TranslateVirtual(addr));
          if (base) {
            for (uint32_t i = 0; i + 24 < rsz / 4; ++i) {
              if (base[i] == 0x30052003u || (base[i] == 0x03200530u && base[i + 1] == 0x00120000u && base[i + 3] == 0x05100000u)) {
                std::string d;
                for (int k = 0; k < 27; ++k) { char t[16]; std::snprintf(t, sizeof(t), "%08X ", base[i + k]); d += t; }
                REXGPU_INFO("GL scan2 hit guest {:08X}: {}", addr + i * 4, d);
                if (++hits >= 8) break;
              }
            }
            total += rsz;
          }
        }
        addr += rsz;
      }
      REXGPU_INFO("GL scan2 done {} MiB hits {}", total >> 20, hits);
    }
  }
  return impl_->pipeline_cache->LoadShader(shader_type, host_address, dword_count);
}

// ---------------------------------------------------------------------------------------------
// Draws
// ---------------------------------------------------------------------------------------------

bool GlCommandProcessor::PrepareVertexBuffers(const Shader& vertex_shader) {
  const RegisterFile& regs = *register_file_;
  const Shader::ConstantRegisterMap& map = vertex_shader.constant_register_map();
  for (uint32_t i = 0; i < rex::countof(map.vertex_fetch_bitmap); ++i) {
    uint32_t bits = map.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(bits, &j)) {
      bits &= ~(uint32_t(1) << j);
      uint32_t index = i * 32 + j;
      xenos::xe_gpu_vertex_fetch_t vfetch = regs.GetVertexFetch(index);
      if (vfetch.type != xenos::FetchConstantType::kVertex &&
          vfetch.type != xenos::FetchConstantType::kInvalidVertex) {
        continue;
      }
      uint32_t address = vfetch.address << 2;
      uint32_t size = vfetch.size << 2;
      if (!size || address >= SharedMemory::kBufferSize) {
        continue;
      }
      size = std::min(size, SharedMemory::kBufferSize - address);
      // No write watches on this platform: re-validate small dynamic buffers by content, larger ones
      // are refreshed once per frame (SetSystemPageBlocksValidWithGpuDataWritten in IssueSwap).
      if (size <= (256u << 10)) {
        uint64_t key = (uint64_t(address) << 32) | size;
        uint64_t hash = XXH3_64bits(memory_->TranslatePhysical(address), size);
        auto it = impl_->vertex_buffer_hashes.find(key);
        if (it == impl_->vertex_buffer_hashes.end() || it->second != hash) {
          impl_->vertex_buffer_hashes[key] = hash;
          impl_->shared_memory->MemoryInvalidationCallback(address, size, true);
        }
      }
      impl_->render_targets->MaterializeLazy(address, size);  // shaders read it from the store
      if (!impl_->shared_memory->RequestRange(address, size)) {
        REXGPU_ERROR("GL: failed to request vertex buffer {:08X} size {} in shared memory", address, size);
        return false;
      }
      if (dbg_log_vertex_) {
        const uint32_t* g = reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(address));
        uint32_t host[8] = {};
        GLuint buf = 0;
        GLintptr boff = 0;
        if (impl_->shared_memory->GetHostRange(address, sizeof(host), buf, boff)) {
          glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
          glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, boff, sizeof(host), host);
        }
        REXGPU_INFO("GL vfetch#{} addr={:08X} size={} guest={:08X} {:08X} {:08X} {:08X} {:08X} {:08X} | host buf={} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                    index, address, size, g[0], g[1], g[2], g[3], g[4], g[5], buf, host[0], host[1], host[2], host[3], host[4], host[5]);
      }
    }
  }
  return true;
}

void GlCommandProcessor::UpdateSystemConstants(bool primitive_polygonal,
                                               const PrimitiveProcessor::ProcessingResult& ppr,
                                               const draw_util::ViewportInfo& viewport_info,
                                               uint32_t used_texture_mask,
                                               reg::RB_DEPTHCONTROL normalized_depth_control) {
  const RegisterFile& regs = *register_file_;
  using ST = SpirvShaderTranslator;
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  auto rb_alpha_ref = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
  auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto vgt_dma_size = regs.Get<reg::VGT_DMA_SIZE>();
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  auto vgt_indx_offset = regs.Get<int32_t>(XE_GPU_REG_VGT_INDX_OFFSET);
  auto vgt_max_vtx_indx = regs.Get<uint32_t>(XE_GPU_REG_VGT_MAX_VTX_INDX);
  auto vgt_min_vtx_indx = regs.Get<uint32_t>(XE_GPU_REG_VGT_MIN_VTX_INDX);
  SpirvShaderTranslator::SystemConstants& sc = system_constants_;

  uint32_t flags = 0;
  if (ppr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA ||
      ppr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA ||
      ppr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted) {
    flags |= ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad;
    if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32) {
      flags |= ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit;
    }
    bool reset = ppr.host_primitive_reset_enabled &&
                 ppr.host_vertex_shader_type != Shader::HostVertexShaderType::kPointListAsTriangleStrip &&
                 ppr.host_vertex_shader_type != Shader::HostVertexShaderType::kRectangleListAsTriangleStrip;
    if (reset) {
      flags |= ST::kSysFlag_ComputeOrPrimitiveVertexIndexReset;
    }
  }
  if (pa_cl_vte_cntl.vtx_xy_fmt) flags |= ST::kSysFlag_XYDividedByW;
  if (pa_cl_vte_cntl.vtx_z_fmt) flags |= ST::kSysFlag_ZDividedByW;
  if (pa_cl_vte_cntl.vtx_w0_fmt) flags |= ST::kSysFlag_WNotReciprocal;
  if (primitive_polygonal) flags |= ST::kSysFlag_PrimitivePolygonal;
  if (draw_util::IsPrimitiveLine(regs)) flags |= ST::kSysFlag_PrimitiveLine;
  flags |= uint32_t(rb_surface_info.msaa_samples) << ST::kSysFlag_MsaaSamples_Shift;
  if (rb_depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
    flags |= ST::kSysFlag_DepthFloat24;
  }
  xenos::CompareFunction alpha_test_function =
      rb_colorcontrol.alpha_test_enable ? rb_colorcontrol.alpha_func : xenos::CompareFunction::kAlways;
  flags |= uint32_t(alpha_test_function) << ST::kSysFlag_AlphaPassIfLess_Shift;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    if (color_info.color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
      flags |= ST::kSysFlag_ConvertColor0ToGamma << i;
    }
  }
  sc.flags = flags;
  sc.vertex_index_load_address = ppr.guest_index_base;
  sc.vertex_index_reset = (flags & ST::kSysFlag_ComputeOrPrimitiveVertexIndexReset)
                              ? regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx
                              : 0;
  sc.compute_memexport_vertex_count = 0;
  xenos::Endian guest_index_endian = vgt_dma_size.swap_mode;
  if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16 &&
      guest_index_endian != xenos::Endian::kNone && guest_index_endian != xenos::Endian::k8in16) {
    guest_index_endian =
        guest_index_endian == xenos::Endian::k8in32 ? xenos::Endian::k8in16 : xenos::Endian::kNone;
  }
  sc.vertex_index_endian = (flags & ST::kSysFlag_ComputeOrPrimitiveVertexIndexLoad)
                               ? guest_index_endian
                               : ppr.host_shader_index_endian;
  sc.line_loop_closing_index = ppr.line_loop_closing_index;
  sc.vertex_base_index = vgt_indx_offset;
  sc.vertex_index_min = vgt_min_vtx_indx;
  sc.vertex_index_max = vgt_max_vtx_indx;

  if (!pa_cl_clip_cntl.clip_disable) {
    float* write_ptr = sc.user_clip_planes[0];
    uint32_t remaining = pa_cl_clip_cntl.ucp_ena;
    uint32_t index;
    while (rex::bit_scan_forward(remaining, &index)) {
      remaining &= ~(UINT32_C(1) << index);
      std::memcpy(write_ptr, &regs[XE_GPU_REG_PA_CL_UCP_0_X + index * 4], 4 * sizeof(float));
      write_ptr += 4;
    }
  }
  sc.tessellation_factor_range_min = regs.Get<float>(XE_GPU_REG_VGT_HOS_MIN_TESS_LEVEL) + 1.0f;
  sc.tessellation_factor_range_max = regs.Get<float>(XE_GPU_REG_VGT_HOS_MAX_TESS_LEVEL) + 1.0f;
  for (uint32_t i = 0; i < 3; ++i) {
    sc.ndc_scale[i] = viewport_info.ndc_scale[i];
    sc.ndc_offset[i] = viewport_info.ndc_offset[i];
  }
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    auto pa_su_point_size = regs.Get<reg::PA_SU_POINT_SIZE>();
    sc.point_vertex_diameter_min = float(pa_su_point_minmax.min_size) * (2.0f / 16.0f);
    sc.point_vertex_diameter_max = float(pa_su_point_minmax.max_size) * (2.0f / 16.0f);
    sc.point_constant_diameter[0] = float(pa_su_point_size.width) * (2.0f / 16.0f);
    sc.point_constant_diameter[1] = float(pa_su_point_size.height) * (2.0f / 16.0f);
    sc.point_screen_diameter_to_ndc_radius[0] = 1.0f / std::max(viewport_info.xy_extent[0], uint32_t(1));
    sc.point_screen_diameter_to_ndc_radius[1] = 1.0f / std::max(viewport_info.xy_extent[1], uint32_t(1));
  }
  sc.textures_resolution_scaled = 0;
  {
    uint32_t remaining = used_texture_mask;
    uint32_t index;
    while (rex::bit_scan_forward(remaining, &index)) {
      remaining &= ~(UINT32_C(1) << index);
      uint32_t& signs_uint = sc.texture_swizzled_signs[index >> 2];
      uint32_t shift = 8 * (index & 3);
      signs_uint = (signs_uint & ~(0xFFu << shift)) | (uint32_t(impl_->fetch_info[index].swizzled_signs) << shift);
      uint32_t& swz_uint = sc.texture_swizzles[index >> 1];
      uint32_t swz_shift = 12 * (index & 1);
      swz_uint = (swz_uint & ~(0xFFFu << swz_shift)) |
                 ((impl_->fetch_info[index].host_swizzle & 0xFFF) << swz_shift);
    }
  }
  sc.alpha_test_reference = rb_alpha_ref;
  sc.alpha_to_mask = rb_colorcontrol.alpha_to_mask_enable ? (rb_colorcontrol.value >> 24) | (UINT32_C(1) << 8) : 0;
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    int32_t color_exp_bias = color_info.color_exp_bias;
    float scale;
    *reinterpret_cast<int32_t*>(&scale) = UINT32_C(0x3F800000) + (color_exp_bias << 23);
    sc.color_exp_bias[i] = scale;
  }
}

void GlCommandProcessor::UploadAndBindConstants(const Shader* vertex_shader, const Shader* pixel_shader) {
  perf::Scope perf_const_scope(perf::kConstUpTicks);
  const RegisterFile& regs = *register_file_;
  auto pack_floats = [&](const Shader& shader, uint32_t reg_base, std::vector<float>& out) {
    const Shader::ConstantRegisterMap& map = shader.constant_register_map();
    out.assign(size_t(std::max<uint32_t>(map.float_count, 1)) * 4, 0.0f);
    float* dst = out.data();
    for (uint32_t i = 0; i < 4; ++i) {
      uint64_t entry = map.float_bitmap[i];
      uint32_t index;
      while (rex::bit_scan_forward(entry, &index)) {
        entry &= ~(1ull << index);
        std::memcpy(dst, &regs[reg_base + (i << 8) + (index << 2)], sizeof(float) * 4);
        dst += 4;
      }
    }
  };
  std::vector<float>& floats_vs = impl_->floats_vs;
  std::vector<float>& floats_ps = impl_->floats_ps;
  pack_floats(*vertex_shader, XE_GPU_REG_SHADER_CONSTANT_000_X, floats_vs);
  if (pixel_shader) {
    pack_floats(*pixel_shader, XE_GPU_REG_SHADER_CONSTANT_256_X, floats_ps);
  } else {
    floats_ps.assign(4, 0.0f);
  }
  struct Item {
    uint32_t binding;
    const void* data;
    uint32_t size;
  };
  const Item items[5] = {
      {kUboBindingSystem, &system_constants_, uint32_t(sizeof(system_constants_))},
      {kUboBindingFloatVertex, floats_vs.data(), uint32_t(floats_vs.size() * 4)},
      {kUboBindingFloatPixel, floats_ps.data(), uint32_t(floats_ps.size() * 4)},
      {kUboBindingBoolLoop, &regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031], uint32_t(sizeof(uint32_t) * (8 + 32))},
      {kUboBindingFetch, &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], uint32_t(sizeof(uint32_t) * 6 * 32)},
  };
  const uint32_t align = std::max<uint32_t>(impl_->ubo_align, 16);
  auto aligned_size = [&](uint32_t size) { return (size + 255) & ~255u; };
  if (impl_->stream) {
    // One contiguous streamed block for all five uniform ranges (never touches memory the GPU still reads).
    uint32_t total = 0;
    for (const Item& it : items) total += (aligned_size(it.size) + align - 1) / align * align;
    uint32_t base = 0;
    uint8_t* dst = static_cast<uint8_t*>(impl_->const_ring.Alloc(total, align, base));
    if (dst) {
      uint32_t off = 0;
      for (const Item& it : items) {
        std::memcpy(dst + off, it.data, it.size);
        glBindBufferRange(GL_UNIFORM_BUFFER, it.binding, impl_->const_ring.buffer(), base + off, aligned_size(it.size));
        off += (aligned_size(it.size) + align - 1) / align * align;
      }
      impl_->const_ring.Commit();
      return;
    }
  }
  for (const Item& it : items) {
    uint32_t aligned = aligned_size(it.size);
    if (impl_->ubo_ring_offset + aligned > Impl::kUboRingSize) {
      glBindBuffer(GL_UNIFORM_BUFFER, impl_->ubo_ring);
      glBufferData(GL_UNIFORM_BUFFER, Impl::kUboRingSize, nullptr, GL_STREAM_DRAW);
      impl_->ubo_ring_offset = 0;
    }
    glBindBuffer(GL_UNIFORM_BUFFER, impl_->ubo_ring);
    glBufferSubData(GL_UNIFORM_BUFFER, impl_->ubo_ring_offset, it.size, it.data);
    glBindBufferRange(GL_UNIFORM_BUFFER, it.binding, impl_->ubo_ring, impl_->ubo_ring_offset, aligned);
    impl_->ubo_ring_offset += aligned;
  }
}

void GlCommandProcessor::ApplyRasterState(bool primitive_polygonal, reg::RB_DEPTHCONTROL depth_control,
                                          uint32_t color_mask,
                                          const draw_util::ViewportInfo& viewport_info,
                                          bool depth_stencil_bound) {
  const RegisterFile& regs = *register_file_;
  RasterStateCache& rc = impl_->raster_cache;
  rc.enabled = REXCVAR_GET(gl_state_cache);
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();

  // Viewport / scissor / depth range (window y is the framebuffer row index; the NDC y flip is done in the
  // translated shader through ndc_scale / ndc_offset with origin_bottom_left).
  if (viewport_info.xy_extent[0] && viewport_info.xy_extent[1]) {
    rc.Viewport(GLint(viewport_info.xy_offset[0]), GLint(viewport_info.xy_offset[1]),
                GLsizei(viewport_info.xy_extent[0]), GLsizei(viewport_info.xy_extent[1]));
  } else {
    rc.Viewport(-1, -1, 1, 1);
  }
  rc.DepthRange(viewport_info.z_min, viewport_info.z_max);
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  rc.Cap(GL_SCISSOR_TEST, true);
  rc.Scissor(GLint(scissor.offset[0]), GLint(scissor.offset[1]), GLsizei(scissor.extent[0]),
             GLsizei(scissor.extent[1]));

  // Rasterizer.
  rc.Cap(GL_DEPTH_CLAMP, regs.Get<reg::PA_CL_CLIP_CNTL>().clip_disable);
  if (primitive_polygonal) {
    bool cull_front = pa_su_sc_mode_cntl.cull_front, cull_back = pa_su_sc_mode_cntl.cull_back;
    if (cull_front || cull_back) {
      rc.Cap(GL_CULL_FACE, true);
      rc.CullFace(cull_front && cull_back ? GL_FRONT_AND_BACK : (cull_front ? GL_FRONT : GL_BACK));
    } else {
      rc.Cap(GL_CULL_FACE, false);
    }
    bool clockwise = pa_su_sc_mode_cntl.face != 0;
    if (REXCVAR_GET(gl_flip_front_face)) {
      clockwise = !clockwise;
    }
    rc.FrontFace(clockwise ? GL_CCW : GL_CW);
    xenos::PolygonType polygon_type = xenos::PolygonType::kTriangles;
    if (!cull_front) polygon_type = std::min(polygon_type, pa_su_sc_mode_cntl.polymode_front_ptype);
    if (!cull_back) polygon_type = std::min(polygon_type, pa_su_sc_mode_cntl.polymode_back_ptype);
    if (pa_su_sc_mode_cntl.poly_mode != xenos::PolygonModeEnable::kDualMode) {
      polygon_type = xenos::PolygonType::kTriangles;
    }
    rc.PolygonMode(polygon_type == xenos::PolygonType::kTriangles ? GL_FILL : GL_LINE);
  } else {
    rc.Cap(GL_CULL_FACE, false);
    rc.PolygonMode(GL_FILL);
  }
  float bias_constant, bias_slope;
  draw_util::GetPreferredFacePolygonOffset(regs, primitive_polygonal, bias_slope, bias_constant);
  bias_constant *= regs.Get<reg::RB_DEPTH_INFO>().depth_format == xenos::DepthRenderTargetFormat::kD24S8
                       ? draw_util::kD3D10PolygonOffsetFactorUnorm24
                       : draw_util::kD3D10PolygonOffsetFactorFloat24;
  bias_slope *= xenos::kPolygonOffsetScaleSubpixelUnit;
  if (bias_constant != 0.0f || bias_slope != 0.0f) {
    rc.Cap(GL_POLYGON_OFFSET_FILL, true);
    rc.Cap(GL_POLYGON_OFFSET_LINE, true);
    rc.Cap(GL_POLYGON_OFFSET_POINT, true);
    rc.PolygonOffset(bias_slope, bias_constant);
  } else {
    rc.Cap(GL_POLYGON_OFFSET_FILL, false);
    rc.Cap(GL_POLYGON_OFFSET_LINE, false);
    rc.Cap(GL_POLYGON_OFFSET_POINT, false);
  }

  // Depth / stencil.
  if (depth_stencil_bound) {
    if (depth_control.z_enable) {
      rc.Cap(GL_DEPTH_TEST, true);
      rc.DepthFunc(CompareFunc(depth_control.zfunc));
      rc.DepthMask(depth_control.z_write_enable);
    } else {
      rc.Cap(GL_DEPTH_TEST, false);
      rc.DepthMask(false);
    }
    if (depth_control.stencil_enable) {
      rc.Cap(GL_STENCIL_TEST, true);
      auto ref_front = regs.Get<reg::RB_STENCILREFMASK>();
      auto ref_back = ref_front;
      bool two_sided = primitive_polygonal && depth_control.backface_enable;
      if (two_sided) {
        ref_back = regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
      }
      rc.StencilSide(0, CompareFunc(depth_control.stencilfunc), ref_front.stencilref, ref_front.stencilmask,
                     StencilOpGl(depth_control.stencilfail), StencilOpGl(depth_control.stencilzfail),
                     StencilOpGl(depth_control.stencilzpass), ref_front.stencilwritemask);
      if (two_sided) {
        rc.StencilSide(1, CompareFunc(depth_control.stencilfunc_bf), ref_back.stencilref, ref_back.stencilmask,
                       StencilOpGl(depth_control.stencilfail_bf), StencilOpGl(depth_control.stencilzfail_bf),
                       StencilOpGl(depth_control.stencilzpass_bf), ref_back.stencilwritemask);
      } else {
        rc.StencilSide(1, CompareFunc(depth_control.stencilfunc), ref_front.stencilref, ref_front.stencilmask,
                       StencilOpGl(depth_control.stencilfail), StencilOpGl(depth_control.stencilzfail),
                       StencilOpGl(depth_control.stencilzpass), ref_front.stencilwritemask);
      }
    } else {
      rc.Cap(GL_STENCIL_TEST, false);
    }
  } else {
    rc.Cap(GL_DEPTH_TEST, false);
    rc.Cap(GL_STENCIL_TEST, false);
  }

  // Blending and color write masks per render target.
  const float blend_color[4] = {regs.Get<float>(XE_GPU_REG_RB_BLEND_RED), regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
                                regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE), regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA)};
  rc.BlendColor(blend_color);
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    uint32_t mask = (color_mask >> (i * 4)) & 0xF;
    rc.ColorMask(i, mask);
    auto blend = regs.Get<reg::RB_BLENDCONTROL>(reg::RB_BLENDCONTROL::rt_register_indices[i]);
    bool blending = mask && !(blend.color_srcblend == xenos::BlendFactor::kOne &&
                              blend.color_destblend == xenos::BlendFactor::kZero &&
                              blend.color_comb_fcn == xenos::BlendOp::kAdd &&
                              blend.alpha_srcblend == xenos::BlendFactor::kOne &&
                              blend.alpha_destblend == xenos::BlendFactor::kZero &&
                              blend.alpha_comb_fcn == xenos::BlendOp::kAdd);
    if (blending) {
      rc.BlendEnable(i, true);
      rc.BlendFunc(i, BlendFactorGl(blend.color_srcblend), BlendFactorGl(blend.color_destblend),
                   BlendFactorGl(blend.alpha_srcblend), BlendFactorGl(blend.alpha_destblend),
                   BlendOpGl(blend.color_comb_fcn), BlendOpGl(blend.alpha_comb_fcn));
    } else {
      rc.BlendEnable(i, false);
    }
  }
}

bool GlCommandProcessor::IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                                   IndexBufferInfo* index_buffer_info, bool major_mode_explicit) {
  const RegisterFile& regs = *register_file_;
  ++draw_count_;
  ++frame_draws_;
  perf::Tick();
  perf::SetCurrentDraw(draw_count_);
  perf::Activity draw_activity(perf::kPhDraw, draw_count_);
  // Per-draw wall time histogram (everything the draw triggers: lookups, uploads, resolves, links).
  struct DrawTimer {
    uint64_t t0 = armGetSystemTick();
    ~DrawTimer() {
      const uint64_t dt = armGetSystemTick() - t0;
      const uint64_t us = armTicksToNs(dt) / 1000;
      perf::Add(us < 1000 ? perf::kDrawHist0 : us < 5000 ? perf::kDrawHist1 : us < 20000 ? perf::kDrawHist2
                : us < 100000 ? perf::kDrawHist3 : us < 1000000 ? perf::kDrawHist4 : perf::kDrawHist5);
      perf::Max(perf::kDrawMaxTicks, dt);
    }
  } draw_timer;
  auto fail = [&](const char* stage) {
    if (++draw_fail_count_ <= 20) {
      REXGPU_ERROR("GL IssueDraw #{} failed at {} (prim={}, indices={})", draw_count_, stage,
                   uint32_t(prim_type), index_count);
    }
    return false;
  };
  xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode == xenos::EdramMode::kCopy) {
    return IssueCopy();
  }
  perf::Add(perf::kDraws);
  perf::Scope perf_draw_scope(perf::kDrawTicks);
  if (!REXCVAR_GET(gl_draw)) {
    return true;
  }
  bool surface_pitch_is_zero = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch == 0;
  GlShader* vertex_shader = static_cast<GlShader*>(active_vertex_shader());
  if (!vertex_shader) {
    return fail("missing_vertex_shader");
  }
  impl_->pipeline_cache->AnalyzeShaderUcode(*vertex_shader);
  bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  bool is_rasterization_done = draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  if (!is_rasterization_done || (surface_pitch_is_zero && is_rasterization_done)) {
    return true;  // nothing but memexport possible
  }
  GlShader* pixel_shader = nullptr;
  if (edram_mode == xenos::EdramMode::kColorDepth) {
    pixel_shader = static_cast<GlShader*>(active_pixel_shader());
    if (pixel_shader) {
      impl_->pipeline_cache->AnalyzeShaderUcode(*pixel_shader);
      if (!draw_util::IsPixelShaderNeededWithRasterization(*pixel_shader, regs)) {
        pixel_shader = nullptr;
      }
    }
  }
  reg::RB_DEPTHCONTROL normalized_depth_control = draw_util::GetNormalizedDepthControl(regs);

  uint32_t ps_param_gen_pos = UINT32_MAX;
  uint32_t interpolator_mask =
      pixel_shader ? (vertex_shader->writes_interpolators() &
                      pixel_shader->GetInterpolatorInputMask(regs.Get<reg::SQ_PROGRAM_CNTL>(),
                                                             regs.Get<reg::SQ_CONTEXT_MISC>(),
                                                             ps_param_gen_pos))
                   : 0;

  // CPU-side "write watch": the guest rewrites index buffers in place; invalidate before processing.
  if (index_buffer_info && index_buffer_info->length) {
    uint32_t base = index_buffer_info->guest_base;
    if (base < SharedMemory::kBufferSize) {
      impl_->shared_memory->MemoryInvalidationCallback(
          base, uint32_t(std::min<size_t>(index_buffer_info->length, SharedMemory::kBufferSize - base)),
          true);
    }
  }
  PrimitiveProcessor::ProcessingResult ppr;
  if (!impl_->primitive_processor->Process(ppr)) {
    return fail("primitive_processing");
  }
  if (!ppr.host_draw_vertex_count) {
    return true;
  }
  if (ppr.IsTessellated()) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXGPU_WARN("GL: tessellated draws are not supported yet (skipped)");
    }
    return true;
  }

  auto vs_mod = impl_->pipeline_cache->GetCurrentVertexShaderModification(
      *vertex_shader, ppr.host_vertex_shader_type, interpolator_mask, ps_param_gen_pos != UINT32_MAX);
  auto ps_mod = pixel_shader ? impl_->pipeline_cache->GetCurrentPixelShaderModification(
                                   *pixel_shader, interpolator_mask, ps_param_gen_pos,
                                   normalized_depth_control)
                             : SpirvShaderTranslator::Modification(0);
  auto* vs_translation =
      static_cast<GlShader::GlTranslation*>(vertex_shader->GetOrCreateTranslation(vs_mod.value));
  auto* ps_translation = pixel_shader ? static_cast<GlShader::GlTranslation*>(
                                            pixel_shader->GetOrCreateTranslation(ps_mod.value))
                                      : nullptr;
  const GlProgram* program = impl_->pipeline_cache->GetProgram(vs_translation, ps_translation);
  if (!program) {
    ++draws_skipped_;
    perf::Add(perf::kSkippedDraws);
    return true;  // still compiling (async) or failed (logged by the pipeline cache)
  }
  const bool first_use = !program->used;
  program->used = true;

  uint32_t normalized_color_mask =
      pixel_shader ? draw_util::GetNormalizedColorMask(regs, pixel_shader->writes_color_targets()) : 0;

  // Viewport and render targets.
  draw_util::ViewportInfo viewport_info;
  draw_util::GetHostViewportInfo(regs, 1, 1, false, uint32_t(REXCVAR_GET(gl_max_viewport)), uint32_t(REXCVAR_GET(gl_max_viewport)), true, normalized_depth_control, false, false,
                                 pixel_shader && pixel_shader->writes_depth(), viewport_info);
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  uint32_t need_w = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch;
  uint32_t vp_bottom = viewport_info.xy_offset[1] + viewport_info.xy_extent[1];
  uint32_t sc_bottom = scissor.offset[1] + scissor.extent[1];
  uint32_t need_h = std::min(std::min(vp_bottom ? vp_bottom : sc_bottom, sc_bottom ? sc_bottom : vp_bottom), 2560u);
  bool depth_stencil_used = normalized_depth_control.z_enable || normalized_depth_control.stencil_enable;
  if (!impl_->render_targets->BindForDraw(normalized_color_mask, depth_stencil_used, need_w, need_h)) {
    return true;  // no render target to draw to
  }

  dbg_log_vertex_ = (REXCVAR_GET(gl_log_draws_from) > 0 && int(draw_count_) >= REXCVAR_GET(gl_log_draws_from) && int(draw_count_) < REXCVAR_GET(gl_log_draws_from) + 12);
  {
    perf::Activity act(perf::kPhVertexPrep);
    if (!PrepareVertexBuffers(*vertex_shader)) {
      return fail("vertex_buffers");
    }
  }

  // Textures (must be bound before the system constants are written: they carry signs/swizzles).
  std::memset(impl_->fetch_info, 0, sizeof(impl_->fetch_info));
  for (auto& fi : impl_->fetch_info) {
    fi.host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }
  const GlslShader* vs_glsl = vs_translation->glsl();
  const GlslShader* ps_glsl = ps_translation ? ps_translation->glsl() : nullptr;
  perf::Activity bind_activity(perf::kPhBindTextures);
  impl_->texture_cache->BindStage(vs_glsl->samplers, vertex_shader->GetTextureBindingsAfterTranslation(),
                                  vertex_shader->GetSamplerBindingsAfterTranslation(), impl_->fetch_info);
  uint32_t used_texture_mask = vertex_shader->GetUsedTextureMaskAfterTranslation();
  if (pixel_shader && ps_glsl) {
    impl_->texture_cache->BindStage(ps_glsl->samplers, pixel_shader->GetTextureBindingsAfterTranslation(),
                                    pixel_shader->GetSamplerBindingsAfterTranslation(), impl_->fetch_info);
    used_texture_mask |= pixel_shader->GetUsedTextureMaskAfterTranslation();
  }

  bool log_this = (REXCVAR_GET(gl_log_draws_from) > 0 && int(draw_count_) >= REXCVAR_GET(gl_log_draws_from) && int(draw_count_) < REXCVAR_GET(gl_log_draws_from) + 12);
  auto chk = [&](const char* what) { if (log_this) { GLenum e = glGetError(); if (e != GL_NO_ERROR) REXGPU_WARN("GL draw #{}: error {} after {}", draw_count_, GlErrorString(e), what); } };
  chk("textures");
  UpdateSystemConstants(primitive_polygonal, ppr, viewport_info, used_texture_mask, normalized_depth_control);
  { static int isp_logged = 0; if (isp_logged++ < 3) REXGPU_INFO("GL: program {} glIsProgram={} (async)", program->program, int(glIsProgram(program->program))); }
  chk("sysconst");
  glUseProgram(program->program);
  if (log_this) { static int up_logged = 0; GLenum e = glGetError(); if (e != GL_NO_ERROR && up_logged++ < 6) { GLint ls = -1, att = -1; glGetProgramiv(program->program, GL_LINK_STATUS, &ls); glGetProgramiv(program->program, GL_ATTACHED_SHADERS, &att); std::string ex; for (GLuint i = 1; i <= 12; ++i) ex += glIsProgram(i) ? "1" : "0"; REXGPU_WARN("GL: glUseProgram({}) error {} link={} attached={} isprog={} exists[1..12]={}", program->program, GlErrorString(e), ls, att, int(glIsProgram(program->program)), ex); } }
  UploadAndBindConstants(vertex_shader, pixel_shader);
  chk("ubo");
  impl_->shared_memory->BindStorageBuffers(kSsboBindingSharedMemory);
  impl_->shared_memory->SetBaseUniform(program->sm_base_location);
  chk("ssbo");
  ApplyRasterState(primitive_polygonal, normalized_depth_control, impl_->render_targets->bound_color_mask()
                                                                      ? normalized_color_mask
                                                                      : 0,
                   viewport_info, impl_->render_targets->bound_depth());

  chk("raster");
  {
    int dbg = REXCVAR_GET(gl_debug_raster);
    if (dbg) impl_->raster_cache.Invalidate();
    if (dbg & 1) glDisable(GL_SCISSOR_TEST);
    if (dbg & 2) glDisable(GL_CULL_FACE);
    if (dbg & 4) { for (uint32_t i = 0; i < 4; ++i) { glColorMaski(i, 1, 1, 1, 1); glDisablei(GL_BLEND, i); } }
    if (dbg & 8) { glDisable(GL_DEPTH_TEST); glDisable(GL_STENCIL_TEST); }
    if (dbg & 16) { glViewport(0, 0, GLsizei(impl_->render_targets->bound_width()), GLsizei(impl_->render_targets->bound_height())); }
  }
  // Primitive topology.
  GLenum mode;
  switch (ppr.host_primitive_type) {
    case xenos::PrimitiveType::kPointList: mode = GL_POINTS; break;
    case xenos::PrimitiveType::kLineList: mode = GL_LINES; break;
    case xenos::PrimitiveType::kLineStrip: mode = GL_LINE_STRIP; break;
    case xenos::PrimitiveType::kTriangleList: mode = GL_TRIANGLES; break;
    case xenos::PrimitiveType::kTriangleStrip: mode = GL_TRIANGLE_STRIP; break;
    case xenos::PrimitiveType::kTriangleFan: mode = GL_TRIANGLE_FAN; break;
    default:
      return fail("unsupported_host_primitive");
  }
  glBindVertexArray(impl_->draw_vao);
  if (ppr.host_primitive_reset_enabled) {
    glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
  } else {
    glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
  }

  if ((REXCVAR_GET(gl_log_draws) > 0 && int(draw_count_) <= REXCVAR_GET(gl_log_draws)) || (REXCVAR_GET(gl_log_draws_from) > 0 && int(draw_count_) >= REXCVAR_GET(gl_log_draws_from) && int(draw_count_) < REXCVAR_GET(gl_log_draws_from) + 12)) {
    REXGPU_INFO(
        "GL draw #{}: prim={}->{} idx={} host_idx={} ibtype={} vs={:016X} ps={:016X} colormask={:x} "
        "rt={}x{} vp=({},{} {}x{}) tex_mask={:x}",
        draw_count_, uint32_t(prim_type), uint32_t(ppr.host_primitive_type), index_count,
        ppr.host_draw_vertex_count, uint32_t(ppr.index_buffer_type), vertex_shader->ucode_data_hash(),
        pixel_shader ? pixel_shader->ucode_data_hash() : uint64_t(0), normalized_color_mask,
        impl_->render_targets->bound_width(), impl_->render_targets->bound_height(), viewport_info.xy_offset[0],
        viewport_info.xy_offset[1], viewport_info.xy_extent[0], viewport_info.xy_extent[1], used_texture_mask);
  }

  if (REXCVAR_GET(gl_debug_raster) & 32) {
    glUseProgram(impl_->dbg_program);
    glBindVertexArray(impl_->empty_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    float px[4] = {-1, -1, -1, -1};
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(100, 100, 1, 1, GL_RGBA, GL_FLOAT, px);
    if (log_this) REXGPU_INFO("GL dbg32 draw #{}: err={} pixel={:.2f} {:.2f} {:.2f} {:.2f}", draw_count_, glGetError(), px[0], px[1], px[2], px[3]);
    return true;
  }
  if (REXCVAR_GET(gl_debug_raster) & 64) {
    glUseProgram(impl_->dbg_program);
  }
  if (REXCVAR_GET(gl_debug_raster) & 128) {
    glUseProgram(impl_->dbg_program);
    glBindVertexArray(impl_->empty_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    float px[4] = {-1, -1, -1, -1};
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(100, 100, 1, 1, GL_RGBA, GL_FLOAT, px);
    if (log_this) REXGPU_INFO("GL dbg128 draw #{}: err={} pixel={:.2f} {:.2f} {:.2f} {:.2f}", draw_count_, glGetError(), px[0], px[1], px[2], px[3]);
    return true;
  }
  using IBT = PrimitiveProcessor::ProcessedIndexBufferType;
  // First draw with a program: Mesa may still build a state-dependent variant here (timed separately).
  perf::Activity first_use_activity(first_use ? perf::kPhFirstUse : perf::kPhDraw,
                                    uint32_t(vertex_shader->ucode_data_hash()),
                                    pixel_shader ? uint32_t(pixel_shader->ucode_data_hash()) : 0);
  const uint64_t first_use_t0 = first_use ? armGetSystemTick() : 0;
  if (ppr.index_buffer_type == IBT::kNone) {
    glDrawArrays(mode, 0, GLsizei(ppr.host_draw_vertex_count));
  } else {
    GLuint buffer = 0;
    GLintptr offset = 0;
    switch (ppr.index_buffer_type) {
      case IBT::kGuestDMA: {
        uint32_t bytes = ppr.host_draw_vertex_count *
                         (ppr.host_index_format == xenos::IndexFormat::kInt16 ? 2u : 4u);
        impl_->render_targets->MaterializeLazy(ppr.guest_index_base, bytes);
        if (!impl_->shared_memory->GetHostRange(ppr.guest_index_base, bytes, buffer, offset)) {
          return fail("index_buffer_not_contiguous");
        }
        break;
      }
      case IBT::kHostConverted:
        buffer = impl_->primitive_processor->UseConvertedIndexBuffer(ppr.host_index_buffer_handle, offset);
        break;
      case IBT::kHostBuiltinForAuto:
      case IBT::kHostBuiltinForDMA:
        buffer = impl_->primitive_processor->builtin_index_buffer();
        offset = impl_->primitive_processor->GetBuiltinOffset(ppr.host_index_buffer_handle);
        break;
      default:
        return fail("index_buffer_type");
    }
    if (!buffer) {
      return fail("index_buffer_null");
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
    glDrawElements(mode, GLsizei(ppr.host_draw_vertex_count),
                   ppr.host_index_format == xenos::IndexFormat::kInt16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT,
                   reinterpret_cast<const void*>(offset));
  }
  if (first_use) {
    perf::Add(perf::kFirstUseDraws);
    perf::Add(perf::kFirstUseTicks, armGetSystemTick() - first_use_t0);
  }
  if (log_this) {
    float px[4] = {-1, -1, -1, -1};
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(100, 100, 1, 1, GL_RGBA, GL_FLOAT, px);
    GLint fbo_bound = 0; glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo_bound);
    REXGPU_INFO("GL draw #{} post: fbo={} pixel(100,100)={:.3f} {:.3f} {:.3f} {:.3f} ndc_scale=({:.5f},{:.5f},{:.5f}) ndc_off=({:.5f},{:.5f},{:.5f}) z=[{:.3f},{:.3f}] sc=({},{} {}x{}) flags={:08X} cullmode={:08X} mask={:x}",
                draw_count_, fbo_bound, px[0], px[1], px[2], px[3], viewport_info.ndc_scale[0], viewport_info.ndc_scale[1], viewport_info.ndc_scale[2],
                viewport_info.ndc_offset[0], viewport_info.ndc_offset[1], viewport_info.ndc_offset[2], viewport_info.z_min, viewport_info.z_max,
                scissor.offset[0], scissor.offset[1], scissor.extent[0], scissor.extent[1], system_constants_.flags,
                regs.Get<reg::PA_SU_SC_MODE_CNTL>().value, normalized_color_mask);
  }
  if ((REXCVAR_GET(gl_log_draws) > 0 && int(draw_count_) <= REXCVAR_GET(gl_log_draws)) || (REXCVAR_GET(gl_log_draws_from) > 0 && int(draw_count_) >= REXCVAR_GET(gl_log_draws_from) && int(draw_count_) < REXCVAR_GET(gl_log_draws_from) + 12)) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
      REXGPU_WARN("GL draw #{}: GL error {}", draw_count_, GlErrorString(err));
    }
  }
  return true;
}

bool GlCommandProcessor::IssueCopy() {
  perf::Tick();
  perf::Activity copy_activity(perf::kPhCopy);
  perf::Add(perf::kCopies);
  perf::Scope perf_copy_scope(perf::kCopyTicks);
  uint32_t start = 0, length = 0;
  glDisable(GL_SCISSOR_TEST);
  bool gpu_written = false;
  bool ok = impl_->render_targets->Resolve(start, length, gpu_written);
  impl_->raster_cache.Invalidate();
  if (length) {
    if (gpu_written) {
      // The store holds the data; guest memory is stale until the lazy write-back.
      impl_->shared_memory->RangeWrittenByGpu(start, length);
    } else {
      impl_->shared_memory->MemoryInvalidationCallback(start, length, true);
    }
  }
  return ok;
}

// ---------------------------------------------------------------------------------------------
// Swap
// ---------------------------------------------------------------------------------------------

void GlCommandProcessor::DrawFpsOverlay(int sw, int sh) {
#if REX_PLATFORM_SWITCH
  if (!REXCVAR_GET(switch_fps_overlay)) return;
  const std::string* lines = perf::OverlayLines();
  if (!lines || lines[0].empty()) return;
  constexpr int kScale = 2, kCell = 6 * kScale, kLineH = 8 * kScale, kPad = 3;
  size_t max_chars = 0;
  int nlines = 0;
  for (int i = 0; i < 3; ++i) {
    if (lines[i].empty()) continue;
    ++nlines;
    max_chars = std::max(max_chars, lines[i].size());
  }
  uint32_t w = uint32_t(max_chars * kCell + 2 * kPad), h = uint32_t(nlines * kLineH + 2 * kPad);
  if (!impl_->overlay_tex[0]) glGenTextures(2, impl_->overlay_tex);
  glActiveTexture(GL_TEXTURE0);
  glBindSampler(0, 0);
  std::string key = lines[0] + "|" + lines[1] + "|" + lines[2];
  if (impl_->overlay_w != w || impl_->overlay_h != h) {
    for (int k = 0; k < 2; ++k) {
      glBindTexture(GL_TEXTURE_2D, impl_->overlay_tex[k]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, GLsizei(w), GLsizei(h), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    impl_->overlay_w = w;
    impl_->overlay_h = h;
    impl_->overlay_text_cached.clear();
  }
  const bool update = key != impl_->overlay_text_cached;
  if (update) impl_->overlay_cur ^= 1;
  glBindTexture(GL_TEXTURE_2D, impl_->overlay_tex[impl_->overlay_cur]);
  if (update) {
    impl_->overlay_text_cached = key;
    std::vector<uint32_t> buf(size_t(w) * h, 0xFF000000u);  // opaque black box, row 0 = top
    int line_y = kPad;
    for (int li = 0; li < 3; ++li) {
      if (lines[li].empty()) continue;
      for (size_t ci = 0; ci < lines[li].size(); ++ci) {
        const uint8_t* g = OverlayGlyph(lines[li][ci]);
        if (!g) continue;
        for (int ry = 0; ry < 7; ++ry) {
          for (int rx = 0; rx < 5; ++rx) {
            if (!((g[ry] >> (4 - rx)) & 1)) continue;
            for (int sy = 0; sy < kScale; ++sy) {
              for (int sx = 0; sx < kScale; ++sx) {
                buf[size_t(line_y + ry * kScale + sy) * w + size_t(kPad + int(ci) * kCell + rx * kScale + sx)] =
                    0xFFFFFFFFu;
              }
            }
          }
        }
      }
      line_y += kLineH;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GLsizei(w), GLsizei(h), GL_RGBA, GL_UNSIGNED_BYTE, buf.data());
  }
  // The swap path already disabled scissor/blend/depth/cull; draw the box into the top-left corner.
  glViewport(0, sh - GLint(h), GLsizei(w), GLsizei(h));
  glUseProgram(impl_->blit_program);
  glBindVertexArray(impl_->empty_vao);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  glViewport(0, 0, sw, sh);
#else
  (void)sw;
  (void)sh;
#endif
}

// Start-up screen while the compile threads load (program binaries) or compile the programs of the program cache. The
// guest has not started yet (g_switch_shaders_ready), so nothing else needs the GPU thread meanwhile.
void GlCommandProcessor::ShaderPrecompileScreen() {
  auto* pc = impl_->pipeline_cache.get();
  const uint32_t total = pc ? pc->warm_total() : 0;
  if (!REXCVAR_GET(gl_precompile_screen) || !total) {
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const int max_s = REXCVAR_GET(gl_precompile_max_s);
  REXGPU_INFO("GL: pre-compiling {} programs before the game starts", total);
  pc->SetStartupPrecompile(true);
  constexpr uint32_t kW = 320, kH = 180;  // canvas, scaled to the whole screen
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glActiveTexture(GL_TEXTURE0);
  glBindSampler(0, 0);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kW, kH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  std::vector<uint32_t> buf(size_t(kW) * kH);
  auto text = [&](const std::string& str, int scale, int y, uint32_t color) {
    const int cell = 6 * scale;
    int x = (int(kW) - int(str.size()) * cell + scale) / 2;
    for (char ch : str) {
      if (const uint8_t* g = OverlayGlyph(ch)) {
        for (int ry = 0; ry < 7; ++ry) {
          for (int rx = 0; rx < 5; ++rx) {
            if (!((g[ry] >> (4 - rx)) & 1)) continue;
            for (int sy = 0; sy < scale; ++sy) {
              for (int sx = 0; sx < scale; ++sx) {
                const int px = x + rx * scale + sx, py = y + ry * scale + sy;
                if (px >= 0 && px < int(kW) && py >= 0 && py < int(kH)) buf[size_t(py) * kW + size_t(px)] = color;
              }
            }
          }
        }
      }
      x += cell;
    }
  };
  auto rect = [&](int x0, int y0, int w, int h, uint32_t color) {
    for (int y = y0; y < y0 + h; ++y) {
      for (int x = x0; x < x0 + w; ++x) buf[size_t(y) * kW + size_t(x)] = color;
    }
  };
  glDisable(GL_SCISSOR_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_STENCIL_TEST);
  glDisable(GL_CULL_FACE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  uint32_t done = 0;
  while (true) {
    done = std::min(pc->warm_done(), total);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool timed_out = max_s > 0 && secs >= max_s;
    // ABGR (GL_RGBA bytes, little endian).
    std::fill(buf.begin(), buf.end(), 0xFF140C08u);
    text("RECOMPILING SHADERS", 2, 52, 0xFFD0E8F0u);
    text(std::to_string(done) + " / " + std::to_string(total), 2, 78, 0xFFFFFFFFu);
    const int bar_w = 240, bar_x = (int(kW) - bar_w) / 2, bar_y = 102;
    rect(bar_x - 2, bar_y - 2, bar_w + 4, 12, 0xFF605048u);
    rect(bar_x, bar_y, bar_w, 8, 0xFF201810u);
    rect(bar_x, bar_y, int(uint64_t(bar_w) * done / total), 8, 0xFF40B0E0u);
    text("PLEASE WAIT", 1, 128, 0xFF8090A0u);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kW, kH, GL_RGBA, GL_UNSIGNED_BYTE, buf.data());
    int sw = 1280, sh = 720;
    context_.QuerySurfaceSize(sw, sh);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, sw, sh);
    glUseProgram(impl_->blit_program);
    glBindVertexArray(impl_->empty_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    context_.SwapBuffers();
    if (done >= total || timed_out) {
      if (timed_out) REXGPU_WARN("GL: pre-compilation stopped after {} s (gl_precompile_max_s)", max_s);
      break;
    }
    svcSleepThread(100'000'000);
  }
  pc->SetStartupPrecompile(false);
  glDeleteTextures(1, &tex);
  glUseProgram(0);
  glBindVertexArray(0);
  impl_->raster_cache.Invalidate();
  REXGPU_INFO("GL: pre-compilation done: {} of {} programs in {:.1f} s ({} from program binaries)", done, total,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), pc->binary_hits());
}

void GlCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                                   uint32_t frontbuffer_height) {
  ++swap_count_;
  perf::Activity swap_activity(perf::kPhSwap, swap_count_);
  perf::OnSwap();
  if (impl_->pipeline_cache) impl_->pipeline_cache->OnFrame();
  perf::Scope perf_swap_scope(perf::kSwapTicks);
  // Sub-step timing: each step() adds the wall time since the previous step to a counter.
  const uint64_t swap_cpu0 = perf::ThreadCpuTicks();
  uint64_t t_step = armGetSystemTick();
  auto step = [&t_step](perf::Counter c) {
    uint64_t n = armGetSystemTick();
    perf::Add(c, n - t_step);
    t_step = n;
  };
  {
    // Guest swaps already queued (VdSwap called) beyond this one: > 0 means the guest is ahead of the GPU thread.
    const uint32_t guest_swaps = g_switch_vdswap_count.load(std::memory_order_relaxed);
    if (guest_swaps > swap_count_) perf::Add(perf::kSwapGuestAhead, guest_swaps - swap_count_);
  }
  perf::GpuSegEnd(perf::kGpuSegFrame, impl_->gpu_frame_begin);
  impl_->gpu_frame_begin = -1;
  perf::GpuLagMark();
  const int gpu_swap_begin = perf::GpuStamp();
#if REX_PLATFORM_SWITCH
  rex::input::switchpad::g_swap_counter.store(swap_count_, std::memory_order_relaxed);
#endif
  // Decode the guest front buffer (fetch constant 0 is set by VdSwap) on the CPU: 32bpp, 32x32 tiled,
  // endian-swapped. Resolves write guest memory on the CPU in this backend, so this is the real image.
  xenos::xe_gpu_texture_fetch_t fetch = register_file_->GetTextureFetch(0);
  uint32_t base = fetch.base_address << 12;
  uint32_t w = fetch.size_2d.width + 1, h = fetch.size_2d.height + 1;
  bool have = base != 0 && w > 1 && h > 1 && w <= 4096 && h <= 4096;
  {
    static uint32_t last_swz = 0xFFFFFFFFu;
    static uint32_t last_fmt = 0xFFFFFFFFu;
    if (uint32_t(fetch.swizzle) != last_swz || uint32_t(fetch.format) != last_fmt) {
      last_swz = uint32_t(fetch.swizzle);
      last_fmt = uint32_t(fetch.format);
      REXGPU_INFO("GL swap #{}: frontbuffer fetch swizzle={:03X} format={} endian={}", swap_count_, last_swz, last_fmt,
                  uint32_t(fetch.endianness));
    }
  }
  if (swap_count_ <= 3 || (swap_count_ % 300) == 0) {
    REXGPU_INFO(
        "GL swap #{}: ptr={:08X} {}x{} fetch base={:08X} {}x{} fmt={} endian={} tiled={} pitch={} | draws={} "
        "total_draws={} failed={} skipped={} shaders={} pending={}",
        swap_count_, frontbuffer_ptr, frontbuffer_width, frontbuffer_height, base, w, h,
        uint32_t(fetch.format), uint32_t(fetch.endianness), uint32_t(fetch.tiled), uint32_t(fetch.pitch) << 5,
        frame_draws_, draw_count_, draw_fail_count_, draws_skipped_, impl_->pipeline_cache->translated_shader_count(), impl_->pipeline_cache->pending_programs());
  }
  frame_draws_ = 0;
  int sw = 1280, sh = 720;
  context_.QuerySurfaceSize(sw, sh);
  // Restore the default (non-draw) state for the blit.
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, sw, sh);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_STENCIL_TEST);
  for (uint32_t i = 0; i < 4; ++i) {
    glDisablei(GL_BLEND, i);
    glColorMaski(i, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  }
  glDisable(GL_CULL_FACE);
  glDisable(GL_SCISSOR_TEST);
  glDisable(GL_DEPTH_CLAMP);
  glDisable(GL_POLYGON_OFFSET_FILL);
  glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  impl_->raster_cache.Invalidate();
  float pulse = float(swap_count_ & 63) / 63.0f;
  glClearColor(0.f, pulse * 0.2f, 0.2f, 1.f);
  step(perf::kSwapPreTicks);
  // First use of the default framebuffer after the previous eglSwapBuffers: Mesa acquires the next back buffer
  // here, so a wait for a free swapchain image (GPU completion / vsync / compositor) shows up in this step.
  glClear(GL_COLOR_BUFFER_BIT);
  step(perf::kSwapClearTicks);
  perf::NotePresentAcquired();
  GLuint snap_tex = 0;
  bool snap_swap_rb = false;
  bool use_snap = false;
  bool use_store = false;
  uint32_t front_ring_offset = 0;
  bool front_from_ring = false;
  if (have) {
    uint32_t pitch = std::max<uint32_t>(uint32_t(fetch.pitch) << 5, w);
    const bool cpu_forced = REXCVAR_GET(gl_cpu_frontbuffer);
    const bool fmt_8888 = fetch.format == xenos::TextureFormat::k_8_8_8_8 || fetch.format == xenos::TextureFormat::k_8_8_8_8_A;
    const bool fmt_1010 = fetch.format == xenos::TextureFormat::k_2_10_10_10;
    const uint32_t swz = uint32_t(fetch.swizzle);
    // Present order: (1) the front buffer is exactly a recent resolve whose GPU snapshot is shown directly (plain
    // textured blit; with lazy resolves the store is never written for it); (2) the front buffer lives in the
    // shared memory store (GPU resolves): a fragment shader detiles it there; (3) CPU detile of guest memory.
    if (!cpu_forced && REXCVAR_GET(gl_present_snapshot) && REXCVAR_GET(gl_resolve_gpu_copy) && fetch.tiled &&
        fmt_8888 && impl_->render_targets->FindSnapshot(base, w, h, pitch, uint32_t(fetch.endianness), 1, snap_tex,
                                                        snap_swap_rb)) {
      use_snap = true;
      perf::Add(perf::kResolveFrontHits);
    }
    if (!use_snap && !cpu_forced && REXCVAR_GET(gl_gpu_front) && impl_->blit_store_program && fetch.tiled &&
        (fmt_8888 || fmt_1010) && uint32_t(fetch.endianness) <= 3) {
      uint32_t upper = texture_util::GetTiledAddressUpperBound2D(w, h, pitch, 2);
      impl_->render_targets->MaterializeLazy(base, upper);
      if (impl_->render_targets->IsGpuResident(base, upper)) {
        use_store = true;
        perf::Add(perf::kGpuFrontBlits);
      } else {
        impl_->render_targets->LogLookupMiss("front", base, w, h, pitch, uint32_t(fetch.endianness),
                                             uint32_t(fetch.format), 1);
      }
    }
    // Old order fallback (gl_present_snapshot = false): snapshot after the store.
    if (!use_store && !use_snap && !cpu_forced && REXCVAR_GET(gl_resolve_gpu_copy) && fetch.tiled && fmt_8888 &&
        impl_->render_targets->FindSnapshot(base, w, h, pitch, uint32_t(fetch.endianness), 1, snap_tex,
                                            snap_swap_rb)) {
      use_snap = true;
      perf::Add(perf::kResolveFrontHits);
    }
    if (!use_store && !use_snap) {
      perf::Add(perf::kCpuFrontFrames);
      const uint32_t pitch_al = (pitch + 31) & ~31u;
      impl_->render_targets->FlushPending(base, pitch_al * ((h + 31) & ~31u) * 4,
                                          GlRenderTargets::FlushReason::kFront);
      const uint8_t* mem = memory_->TranslatePhysical(base);
      perf::Scope perf_detile_scope(perf::kSwapDetileTicks);
      uint32_t* out_px = nullptr;
      size_t bytes = size_t(w) * h * 4;
      if (impl_->stream && bytes <= impl_->tex_ring.segment_size()) {
        out_px = static_cast<uint32_t*>(impl_->tex_ring.Alloc(uint32_t(bytes), 64, front_ring_offset));
        front_from_ring = out_px != nullptr;
      }
      if (!front_from_ring) {
        impl_->front_pixels.resize(size_t(w) * h);
        out_px = impl_->front_pixels.data();
      }
      const bool need_swz = (swz & 0x1FF) != 0x88;
      const bool tiled = fetch.tiled;
      const xenos::Endian endian = fetch.endianness;
      uint32_t any_nonzero = 0;
      for (uint32_t y = 0; y < h; ++y) {
        uint32_t* drow = out_px + size_t(y) * w;
        for (uint32_t x = 0; x < w; ++x) {
          uint32_t off = tiled ? TiledOffset2D(x, y, pitch_al, 2) : (y * pitch + x) * 4;
          uint32_t v;
          std::memcpy(&v, mem + off, 4);
          switch (endian) {
            case xenos::Endian::k8in32: v = __builtin_bswap32(v); break;
            case xenos::Endian::k8in16: v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu); break;
            case xenos::Endian::k16in32: v = (v << 16) | (v >> 16); break;
            default: break;
          }
          if (need_swz) {
            // Apply the fetch-constant swizzle to the colour channels.
            uint32_t comp[6] = {v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF, 0, 0xFF};
            uint32_t r = comp[std::min<uint32_t>(swz & 7, 5)], g = comp[std::min<uint32_t>((swz >> 3) & 7, 5)],
                     b = comp[std::min<uint32_t>((swz >> 6) & 7, 5)];
            v = r | (g << 8) | (b << 16) | (v & 0xFF000000u);
          }
          any_nonzero |= v;
          drow[x] = v;
        }
      }
      if (front_from_ring) impl_->tex_ring.Commit();
      if (!any_nonzero) {
        have = false;
      }
    }
  }
  if (have) {
    glActiveTexture(GL_TEXTURE0);
    if (use_store) {
      const uint32_t swz = uint32_t(fetch.swizzle);
      impl_->shared_memory->BindStorageBuffers(0);
      glUseProgram(impl_->blit_store_program);
      glUniform4ui(impl_->blit_store_a_loc, base, (std::max<uint32_t>(uint32_t(fetch.pitch) << 5, w) + 31) & ~31u, w, h);
      glUniform4i(impl_->blit_store_b_loc, GLint(uint32_t(fetch.endianness)), fetch.format == xenos::TextureFormat::k_2_10_10_10 ? 1 : 0, 0, 0);
      glUniform3i(impl_->blit_store_sel_loc, GLint(swz & 7), GLint((swz >> 3) & 7), GLint((swz >> 6) & 7));
    } else if (use_snap) {
      glBindTexture(GL_TEXTURE_2D_ARRAY, snap_tex);
      glBindSampler(0, 0);
      glUseProgram(impl_->blit_array_program);
      // Channel selector: fetch swizzle applied to the guest bytes; the snapshot is the render target (RGBA) and
      // the resolve may have swapped red/blue on the way into guest memory.
      const uint32_t swz = uint32_t(fetch.swizzle);
      auto sel = [&](uint32_t s) -> GLint {
        s &= 7;
        if (s < 4 && snap_swap_rb && (s == 0 || s == 2)) return GLint(2 - s);
        return GLint(s);
      };
      glUniform3i(impl_->blit_array_sel_loc, sel(swz), sel(swz >> 3), sel(swz >> 6));
    } else {
      glBindTexture(GL_TEXTURE_2D, impl_->front_tex);
      if (impl_->front_w != w || impl_->front_h != h) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        impl_->front_w = w;
        impl_->front_h = h;
      }
      glBindSampler(0, 0);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
      if (front_from_ring) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, impl_->tex_ring.buffer());
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE,
                        reinterpret_cast<const void*>(uintptr_t(front_ring_offset)));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, impl_->front_pixels.data());
      }
      glUseProgram(impl_->blit_program);
    }
    glBindVertexArray(impl_->empty_vao);
    const int gpu_blit_begin = perf::GpuStamp();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    perf::GpuSegEnd(perf::kGpuSegBlit, gpu_blit_begin);
  }
  step(perf::kSwapBlitTicks);
  std::string dir = REXCVAR_GET(gl_screenshot_dir);
  if (!dir.empty()) {
    int every = REXCVAR_GET(gl_screenshot_swaps);
    int first = REXCVAR_GET(gl_screenshot_first);
    int from = REXCVAR_GET(gl_screenshot_from), from_every = std::max(1, REXCVAR_GET(gl_screenshot_from_every));
    bool shot = (first && int(swap_count_) == first) || (every > 0 && (swap_count_ % every) == 0) ||
                (from > 0 && int(swap_count_) >= from && (swap_count_ % from_every) == 0);
    if (shot) {
      WritePpm(dir + "/shot_" + std::to_string(swap_count_) + ".ppm", sw, sh);
    }
  }
  {
    int every_rt = REXCVAR_GET(gl_dump_rt_swaps);
    std::string rdir = REXCVAR_GET(gl_screenshot_dir);
    if (every_rt > 0 && !rdir.empty() && (swap_count_ % every_rt) == 0) {
      std::vector<uint8_t> rgba;
      uint32_t rw = 0, rh = 0;
      if (impl_->render_targets->DebugReadColor(0, rgba, rw, rh)) {
        std::string path = rdir + "/rt_" + std::to_string(swap_count_) + ".ppm";
        if (FILE* f = fopen(path.c_str(), "wb")) {
          fprintf(f, "P6\n%u %u\n255\n", rw, rh);
          std::vector<uint8_t> row(size_t(rw) * 3);
          for (uint32_t y = 0; y < rh; ++y) {
            for (uint32_t x = 0; x < rw; ++x) {
              for (int k = 0; k < 3; ++k) row[x * 3 + k] = rgba[(size_t(y) * rw + x) * 4 + k];
            }
            fwrite(row.data(), 1, row.size(), f);
          }
          fclose(f);
        }
      }
    }
  }
  step(perf::kSwapMiscTicks);
  DrawFpsOverlay(sw, sh);
  step(perf::kSwapOverlayTicks);
  perf::GpuSegEnd(perf::kGpuSegSwap, gpu_swap_begin);
  perf::GpuFrameEndMark();
  {
    perf::Scope perf_swapbuf_scope(perf::kSwapBufTicks);
    context_.SwapBuffers();
  }
  t_step = armGetSystemTick();
  perf::FrameFenceIssue();
  impl_->gpu_frame_begin = perf::GpuStamp();

  // Per-frame housekeeping.
  impl_->shared_memory->SetSystemPageBlocksValidWithGpuDataWritten();
  impl_->primitive_processor->EndFrame();
  impl_->texture_cache->EndFrame();
  impl_->render_targets->PollPending();
  impl_->render_targets->EndFrame();
  impl_->vertex_buffer_hashes.clear();
if (!impl_->stream) {    impl_->ubo_ring_offset = 0;    glBindBuffer(GL_UNIFORM_BUFFER, impl_->ubo_ring);    glBufferData(GL_UNIFORM_BUFFER, Impl::kUboRingSize, nullptr, GL_STREAM_DRAW);  }
  perf::GpuPoll();
  step(perf::kSwapPostTicks);
  const uint64_t swap_cpu1 = perf::ThreadCpuTicks();
  if (swap_cpu0 && swap_cpu1 > swap_cpu0) perf::Add(perf::kSwapCpuTicks, swap_cpu1 - swap_cpu0);
}

}  // namespace rex::graphics::gl
