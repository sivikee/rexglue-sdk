#include "gl_context.h"
#include "gl_ext.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <glad/glad.h>
#include <switch.h>

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(gl_egl_depth_stencil, false, "GPU",
                    "GL backend: request depth 24 / stencil 8 for the EGL window surface (the default framebuffer is only "
                    "cleared and blitted to; render targets are FBOs)");

REXCVAR_DEFINE_INT32(gl_nvc0_code_area_mb, 8, "GPU",
                     "GL backend: grow nouveau's shader code area to this size (MiB, max 8) right after start-up. Mesa "
                     "starts with 512 KiB and, when the translated in-game shaders do not fit, evicts every shader and "
                     "reallocates it in the middle of a frame (suspected cause of an Eden hang). 0 = leave as is");
extern "C" int nvc0_screen_resize_text_area(void* screen, uint64_t size);
extern "C" void nvc0_program_library_upload(void* nvc0_context);

REXCVAR_DEFINE_STRING(gl_mesa_env, "", "GPU",
                      "GL backend: environment variables for Mesa, set before EGL starts, as NAME=VALUE pairs separated "
                      "by ';' (e.g. \"MESA_GLSL=nopt;NV50_PROG_OPTIMIZE=1\")");

namespace rex::graphics::gl {

const char* GlErrorString(unsigned int err) {
  switch (err) {
    case GL_NO_ERROR: return "GL_NO_ERROR";
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
    default: return "GL_?";
  }
}

extern "C" void oblivion_bootlog(const char* fmt, ...) __attribute__((weak, format(printf, 1, 2)));
#define GLBL(...) do { if (oblivion_bootlog) oblivion_bootlog(__VA_ARGS__); } while (0)

bool GlContext::Initialize() {
  {
    // Mesa reads these when the display / context / compiler start (MESA_GLSL at context creation,
    // NV50_PROG_OPTIMIZE per shader).
    const std::string env = REXCVAR_GET(gl_mesa_env);
    size_t pos = 0;
    while (pos < env.size()) {
      size_t end = env.find(';', pos);
      if (end == std::string::npos) end = env.size();
      const std::string item = env.substr(pos, end - pos);
      const size_t eq = item.find('=');
      if (eq != std::string::npos && eq > 0) {
        setenv(item.substr(0, eq).c_str(), item.substr(eq + 1).c_str(), 1);
        REXGPU_INFO("GL: Mesa environment {}", item);
      }
      pos = end + 1;
    }
  }
  GLBL("gl: eglGetDisplay");
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (!display) {
    REXGPU_ERROR("GL: eglGetDisplay failed ({:x})", eglGetError());
    return false;
  }
  GLBL("gl: eglInitialize");
  EGLint major = 0, minor = 0;
  if (!eglInitialize(display, &major, &minor)) {
    REXGPU_ERROR("GL: eglInitialize failed ({:x})", eglGetError());
    return false;
  }
  GLBL("gl: eglBindAPI");
  if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
    REXGPU_ERROR("GL: eglBindAPI(OPENGL) failed ({:x})", eglGetError());
    return false;
  }
  const bool want_ds = REXCVAR_GET(gl_egl_depth_stencil);
  const EGLint fb[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
                       EGL_RED_SIZE,        8,              EGL_GREEN_SIZE,   8,
                       EGL_BLUE_SIZE,       8,              EGL_ALPHA_SIZE,   8,
                       EGL_DEPTH_SIZE,      want_ds ? 24 : 0, EGL_STENCIL_SIZE, want_ds ? 8 : 0,
                       EGL_SAMPLE_BUFFERS,  0,              EGL_SAMPLES,      0,
                       EGL_NONE};
  GLBL("gl: eglChooseConfig");
  EGLConfig config;
  EGLint num = 0;
  eglChooseConfig(display, fb, &config, 1, &num);
  if (!num) {
    REXGPU_ERROR("GL: no EGL config ({:x})", eglGetError());
    return false;
  }
  {
    EGLint r = 0, g = 0, b = 0, a = 0, d = 0, st = 0, sm = 0, vid = 0;
    eglGetConfigAttrib(display, config, EGL_RED_SIZE, &r);
    eglGetConfigAttrib(display, config, EGL_GREEN_SIZE, &g);
    eglGetConfigAttrib(display, config, EGL_BLUE_SIZE, &b);
    eglGetConfigAttrib(display, config, EGL_ALPHA_SIZE, &a);
    eglGetConfigAttrib(display, config, EGL_DEPTH_SIZE, &d);
    eglGetConfigAttrib(display, config, EGL_STENCIL_SIZE, &st);
    eglGetConfigAttrib(display, config, EGL_SAMPLES, &sm);
    eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &vid);
    GLBL("gl: EGL config rgba=%d%d%d%d depth=%d stencil=%d samples=%d visual=0x%x (depth/stencil requested: %d)", r, g, b,
         a, d, st, sm, vid, int(want_ds));
  }
  GLBL("gl: eglCreateWindowSurface");
  EGLSurface surface = eglCreateWindowSurface(display, config, nwindowGetDefault(), nullptr);
  if (!surface) {
    REXGPU_ERROR("GL: eglCreateWindowSurface failed ({:x})", eglGetError());
    return false;
  }
  EGLint attr[] = {EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
                   EGL_CONTEXT_MAJOR_VERSION_KHR,       4,
                   EGL_CONTEXT_MINOR_VERSION_KHR,       3,
                   EGL_NONE};
  GLBL("gl: eglCreateContext");
  EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, attr);
  if (!context) {
    REXGPU_ERROR("GL: eglCreateContext(core 4.3) failed ({:x})", eglGetError());
    return false;
  }
  GLBL("gl: eglMakeCurrent");
  if (!eglMakeCurrent(display, surface, surface, context)) {
    REXGPU_ERROR("GL: eglMakeCurrent failed ({:x})", eglGetError());
    return false;
  }
  GLBL("gl: EGL context ready");
  config_ = config;
  display_ = display;
  surface_ = surface;
  context_ = context;
  if (!gladLoadGLLoader((GLADloadproc)eglGetProcAddress)) {
    REXGPU_ERROR("GL: gladLoadGLLoader failed");
    return false;
  }
  LoadGlExtensions([](const char* n) -> void* { return reinterpret_cast<void*>(eglGetProcAddress(n)); });
  auto str = [](GLenum n) {
    const GLubyte* s = glGetString(n);
    return s ? std::string(reinterpret_cast<const char*>(s)) : std::string();
  };
  caps_.renderer = str(GL_RENDERER);
  caps_.version = str(GL_VERSION);
  caps_.glsl_version = str(GL_SHADING_LANGUAGE_VERSION);
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &caps_.max_texture_size);
  glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &caps_.max_ssbo_block_size);
  glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &caps_.max_uniform_block_size);
  glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &caps_.max_texture_units);
  glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &caps_.max_color_attachments);
  GLint n = 0;
  bool have_buffer_storage = false;
  glGetIntegerv(GL_NUM_EXTENSIONS, &n);
  for (GLint i = 0; i < n; ++i) {
    const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));
    if (!e) continue;
    if (!strcmp(e, "GL_EXT_texture_filter_anisotropic")) caps_.ext_texture_filter_anisotropic = true;
    if (!strcmp(e, "GL_ARB_cull_distance")) caps_.arb_cull_distance = true;
    if (!strcmp(e, "GL_ARB_buffer_storage")) have_buffer_storage = true;
    if (!strcmp(e, "GL_EXT_demote_to_helper_invocation")) caps_.ext_demote_to_helper = true;
    if (!strcmp(e, "GL_EXT_texture_compression_s3tc")) caps_.s3tc = true;
  }
  int w = 0, h = 0;
  if (!have_buffer_storage) glBufferStorage_ = nullptr;
  REXGPU_INFO("GL: ARB_buffer_storage {}", have_buffer_storage ? "available" : "missing");
  QuerySurfaceSize(w, h);
  REXGPU_INFO("GL: EGL {}.{} renderer='{}' version='{}' glsl='{}' surface={}x{} max_tex={} max_ssbo={}",
              major, minor, caps_.renderer, caps_.version, caps_.glsl_version, w, h,
              caps_.max_texture_size, caps_.max_ssbo_block_size);
  eglSwapInterval(display, 1);
  GrowShaderCodeArea();
  return true;
}

// Mesa 20.1 nvc0 (devkitPro build, statically linked): struct st_manager starts with its pipe_screen*; in the
// pipe_screen (= nvc0_screen) get_name() is at +16 (+0 num_contexts, +8 destroy; st_get_string(GL_RENDERER) loads
// [screen + 16]); nvc0_screen::text (the code segment BO) is at
// +632 and nouveau_bo::size at +16 (disassembly of nvc0_screen_resize_text_area / nvc0_program_upload). Checked:
// get_name() must return the GL_RENDERER string and the code area must still have its initial 512 KiB.
void GlContext::GrowShaderCodeArea() {
  const int mb = std::min(REXCVAR_GET(gl_nvc0_code_area_mb), 8);
  if (mb <= 0 || !display_) return;
  void** driver_data = *reinterpret_cast<void***>(static_cast<char*>(display_) + 88);
  void* smapi = driver_data ? driver_data[0] : nullptr;
  void* screen = smapi ? *reinterpret_cast<void**>(smapi) : nullptr;
  if (!screen) {
    REXGPU_WARN("GL: shader code area left as is (no pipe screen found)");
    return;
  }
  using GetNameFn = const char* (*)(void*);
  GetNameFn get_name = *reinterpret_cast<GetNameFn*>(static_cast<char*>(screen) + 16);
  const char* name = get_name ? get_name(screen) : nullptr;
  if (!name || caps_.renderer != name) {
    REXGPU_WARN("GL: shader code area left as is (screen layout check failed: name '{}' vs renderer '{}')",
                name ? name : "?", caps_.renderer);
    return;
  }
  void* text = *reinterpret_cast<void**>(static_cast<char*>(screen) + 632);
  const uint64_t size = text ? *reinterpret_cast<uint64_t*>(static_cast<char*>(text) + 16) : 0;
  const uint64_t want = uint64_t(mb) << 20;
  // Only the builtin library may live in the code heap (uploaded by nvc0_create): nvc0_screen::text_heap at +0x2A8
  // is the root node {prev, next, priv, start, size, in_use} whose next must be the library node (+0x2B0, priv NULL,
  // no next). The resize frees both nodes, so the library is uploaded again through the main context
  // (st_context::pipe at +104 of the main context's st_context, checked via pipe_context::screen at +0).
  void* heap = *reinterpret_cast<void**>(static_cast<char*>(screen) + 0x2A8);
  void* lib = *reinterpret_cast<void**>(static_cast<char*>(screen) + 0x2B0);
  auto field = [](void* node, size_t off) { return *reinterpret_cast<void**>(static_cast<char*>(node) + off); };
  const bool heap_ok = heap && lib && field(heap, 8) == lib && !field(lib, 8) && !field(lib, 16) &&
                       *reinterpret_cast<int*>(static_cast<char*>(lib) + 32) == 1;
  void* st = context_ ? *reinterpret_cast<void**>(static_cast<char*>(context_) + 104) : nullptr;
  void* pipe = st ? *reinterpret_cast<void**>(static_cast<char*>(st) + 104) : nullptr;
  const bool pipe_ok = pipe && *reinterpret_cast<void**>(pipe) == screen;
  if (size != 0x80000 || want <= size || !heap_ok || !pipe_ok) {
    REXGPU_WARN("GL: shader code area left as is (size 0x{:X}, wanted 0x{:X}, heap ok {}, pipe ok {})", size, want,
                heap_ok, pipe_ok);
    return;
  }
  const int rc = nvc0_screen_resize_text_area(screen, want);
  if (rc == 0) nvc0_program_library_upload(pipe);
  void* lib_after = *reinterpret_cast<void**>(static_cast<char*>(screen) + 0x2B0);
  REXGPU_INFO("GL: shader code area grown from {} KiB to {} MiB (rc {}, library re-uploaded {})", size >> 10, mb, rc,
              lib_after != nullptr);
}

void GlContext::Shutdown() {
  if (display_) {
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (context_) eglDestroyContext(display_, context_);
    if (surface_) eglDestroySurface(display_, surface_);
    eglTerminate(display_);
  }
  display_ = surface_ = context_ = nullptr;
}

void GlContext::SetSwapInterval(int interval) {
  if (display_) eglSwapInterval(static_cast<EGLDisplay>(display_), interval);
}

void GlContext::SwapBuffers() {
  eglSwapBuffers(static_cast<EGLDisplay>(display_), static_cast<EGLSurface>(surface_));
}

void GlContext::QuerySurfaceSize(int& width, int& height) const {
  EGLint w = 1280, h = 720;
  if (display_ && surface_) {
    eglQuerySurface(static_cast<EGLDisplay>(display_), static_cast<EGLSurface>(surface_), EGL_WIDTH, &w);
    eglQuerySurface(static_cast<EGLDisplay>(display_), static_cast<EGLSurface>(surface_), EGL_HEIGHT, &h);
  }
  if (w <= 0 || h <= 0) {
    u32 nw = 0, nh = 0;
    nwindowGetDimensions(nwindowGetDefault(), &nw, &nh);
    w = nw ? EGLint(nw) : 1280;
    h = nh ? EGLint(nh) : 720;
  }
  width = w;
  height = h;
}

}  // namespace rex::graphics::gl

namespace rex::graphics::gl {
void (*glBufferStorage_)(GLenum, GLsizeiptr, const void*, GLbitfield) = nullptr;
void (*glClipControl_)(GLenum, GLenum) = nullptr;
void (*glTextureBarrier_)() = nullptr;
void (*glPolygonOffsetClamp_)(GLfloat, GLfloat, GLfloat) = nullptr;
void (*glMaxShaderCompilerThreadsKHR_)(GLuint) = nullptr;

void LoadGlExtensions(void* (*get_proc_address)(const char*)) {
  glMaxShaderCompilerThreadsKHR_ = reinterpret_cast<decltype(glMaxShaderCompilerThreadsKHR_)>(get_proc_address("glMaxShaderCompilerThreadsKHR"));
  glBufferStorage_ = reinterpret_cast<decltype(glBufferStorage_)>(get_proc_address("glBufferStorage"));
  glClipControl_ = reinterpret_cast<decltype(glClipControl_)>(get_proc_address("glClipControl"));
  glTextureBarrier_ = reinterpret_cast<decltype(glTextureBarrier_)>(get_proc_address("glTextureBarrier"));
  glPolygonOffsetClamp_ =
      reinterpret_cast<decltype(glPolygonOffsetClamp_)>(get_proc_address("glPolygonOffsetClampEXT"));
}
}  // namespace rex::graphics::gl

// devkitPro's Mesa 20.1 Switch EGL driver (egl_switch.c, statically linked from libEGL.a) IGNORES the share_context
// argument of eglCreateContext: its switch_create_context calls st_api::create_context(..., shared_stctx = NULL)
// (disassembly: `mov x4, #0` before `blr` of st_api+40; the share_list argument is never even saved). A second
// context therefore never shares programs / buffers / syncs with the first one, which is why the v0.1.12/v0.1.13
// compile thread could not hand its programs over ("share probe: buffer false program false sync false").
// Workaround (gl_egl_share_fix): while the shared context is created, the display's st_api pointer is temporarily
// replaced by a copy whose create_context injects the main context's st_context_iface as shared_stctx. All offsets
// are taken from that disassembly and verified at run time before anything is patched (st_api pointer ==
// st_gl_api_create(), st_api::get_current() == the main context's stctx); on any mismatch the plain call is used.
REXCVAR_DEFINE_BOOL(gl_egl_share_fix, true, "GPU",
                    "GL backend: make eglCreateContext's share_context work on the Switch EGL driver (it is ignored by "
                    "devkitPro Mesa 20.1) so the compile thread's shared context really shares programs");

extern "C" void* st_gl_api_create(void);

namespace {
constexpr size_t kEglDisplayDriverDataOffset = 88;  // _EGLDisplay::DriverData (switch_create_context: ldr x0,[x1,#88])
constexpr size_t kSwitchCtxStctxOffset = 104;       // switch context: st_context_iface* (str x0,[x19,#104])
constexpr size_t kStApiCreateContext = 40;          // st_api::create_context (ldr x5,[x0,#40])
constexpr size_t kStApiGetCurrent = 56;             // st_api::get_current
constexpr size_t kStApiSize = 72;                   // sizeof(struct st_api), Mesa 20.1
using StCreateContextFn = void* (*)(void* api, void* smapi, const void* attribs, int* error, void* shared);
using StGetCurrentFn = void* (*)(void* api);
StCreateContextFn g_orig_create_context = nullptr;
void* g_orig_st_api = nullptr;
void* g_share_stctx = nullptr;
alignas(16) unsigned char g_st_api_copy[128];
void* SharingCreateContext(void* /*api*/, void* smapi, const void* attribs, int* error, void* shared) {
  if (!shared && g_share_stctx) shared = g_share_stctx;
  return g_orig_create_context(g_orig_st_api, smapi, attribs, error, shared);
}
}  // namespace

namespace rex::graphics::gl {
void* GlContext::CreateSharedContext() {
  EGLint attr[] = {EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
                   EGL_CONTEXT_MAJOR_VERSION_KHR,       4,
                   EGL_CONTEXT_MINOR_VERSION_KHR,       3,
                   EGL_NONE};
  void** driver_data = nullptr;
  void* st_api = nullptr;
  if (REXCVAR_GET(gl_egl_share_fix) && display_ && context_) {
    driver_data = *reinterpret_cast<void***>(static_cast<char*>(display_) + kEglDisplayDriverDataOffset);
    st_api = driver_data ? driver_data[1] : nullptr;
    void* main_stctx = *reinterpret_cast<void**>(static_cast<char*>(context_) + kSwitchCtxStctxOffset);
    void* expected_api = st_gl_api_create();
    void* current = nullptr;
    if (st_api == expected_api && main_stctx) {
      auto get_current = *reinterpret_cast<StGetCurrentFn*>(static_cast<char*>(st_api) + kStApiGetCurrent);
      current = get_current ? get_current(st_api) : nullptr;  // the GPU thread's context is current here
    }
    if (st_api && st_api == expected_api && main_stctx && current == main_stctx) {
      std::memcpy(g_st_api_copy, st_api, kStApiSize);
      g_orig_st_api = st_api;
      g_orig_create_context = *reinterpret_cast<StCreateContextFn*>(static_cast<char*>(st_api) + kStApiCreateContext);
      *reinterpret_cast<StCreateContextFn*>(g_st_api_copy + kStApiCreateContext) = &SharingCreateContext;
      g_share_stctx = main_stctx;
      driver_data[1] = g_st_api_copy;
      REXGPU_INFO("GL: EGL share fix armed (st_api {}, main st context {})", st_api, main_stctx);
    } else {
      REXGPU_WARN("GL: EGL share fix NOT applied (layout check failed: driver data {}, st_api {} vs st_gl_api_create {}, "
                  "main st context {}, current {}); the shared context will not share objects",
                  static_cast<void*>(driver_data), st_api, expected_api, main_stctx, current);
      driver_data = nullptr;
    }
  }
  EGLContext ctx = eglCreateContext(static_cast<EGLDisplay>(display_), static_cast<EGLConfig>(config_),
                                    static_cast<EGLContext>(context_), attr);
  if (driver_data) {
    driver_data[1] = st_api;  // restore: nothing else ever sees the copy
    g_share_stctx = nullptr;
  }
  if (!ctx) {
    REXGPU_ERROR("GL: shared context creation failed ({:x})", eglGetError());
  }
  return ctx;
}
bool GlContext::MakeCurrentWorker(void* shared_context) {
  return eglMakeCurrent(static_cast<EGLDisplay>(display_), EGL_NO_SURFACE, EGL_NO_SURFACE,
                        static_cast<EGLContext>(shared_context)) == EGL_TRUE;
}
void GlContext::ReleaseCurrentWorker() {
  eglMakeCurrent(static_cast<EGLDisplay>(display_), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}
}  // namespace rex::graphics::gl
