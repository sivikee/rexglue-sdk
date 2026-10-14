#include "gl_context.h"
#include "gl_ext.h"

#include <cstring>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <glad/glad.h>
#include <switch.h>

#include <rex/logging.h>

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

bool GlContext::Initialize() {
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (!display) {
    REXGPU_ERROR("GL: eglGetDisplay failed ({:x})", eglGetError());
    return false;
  }
  EGLint major = 0, minor = 0;
  if (!eglInitialize(display, &major, &minor)) {
    REXGPU_ERROR("GL: eglInitialize failed ({:x})", eglGetError());
    return false;
  }
  if (eglBindAPI(EGL_OPENGL_API) == EGL_FALSE) {
    REXGPU_ERROR("GL: eglBindAPI(OPENGL) failed ({:x})", eglGetError());
    return false;
  }
  static const EGLint fb[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_RED_SIZE,     8,
                              EGL_GREEN_SIZE,      8,               EGL_BLUE_SIZE,    8,
                              EGL_ALPHA_SIZE,      8,               EGL_DEPTH_SIZE,   24,
                              EGL_STENCIL_SIZE,    8,               EGL_NONE};
  EGLConfig config;
  EGLint num = 0;
  eglChooseConfig(display, fb, &config, 1, &num);
  if (!num) {
    REXGPU_ERROR("GL: no EGL config ({:x})", eglGetError());
    return false;
  }
  EGLSurface surface = eglCreateWindowSurface(display, config, nwindowGetDefault(), nullptr);
  if (!surface) {
    REXGPU_ERROR("GL: eglCreateWindowSurface failed ({:x})", eglGetError());
    return false;
  }
  EGLint attr[] = {EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
                   EGL_CONTEXT_MAJOR_VERSION_KHR,       4,
                   EGL_CONTEXT_MINOR_VERSION_KHR,       3,
                   EGL_NONE};
  EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, attr);
  if (!context) {
    REXGPU_ERROR("GL: eglCreateContext(core 4.3) failed ({:x})", eglGetError());
    return false;
  }
  if (!eglMakeCurrent(display, surface, surface, context)) {
    REXGPU_ERROR("GL: eglMakeCurrent failed ({:x})", eglGetError());
    return false;
  }
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
  glGetIntegerv(GL_NUM_EXTENSIONS, &n);
  for (GLint i = 0; i < n; ++i) {
    const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));
    if (!e) continue;
    if (!strcmp(e, "GL_EXT_texture_filter_anisotropic")) caps_.ext_texture_filter_anisotropic = true;
    if (!strcmp(e, "GL_ARB_cull_distance")) caps_.arb_cull_distance = true;
    if (!strcmp(e, "GL_EXT_demote_to_helper_invocation")) caps_.ext_demote_to_helper = true;
    if (!strcmp(e, "GL_EXT_texture_compression_s3tc")) caps_.s3tc = true;
  }
  int w = 0, h = 0;
  QuerySurfaceSize(w, h);
  REXGPU_INFO("GL: EGL {}.{} renderer='{}' version='{}' glsl='{}' surface={}x{} max_tex={} max_ssbo={}",
              major, minor, caps_.renderer, caps_.version, caps_.glsl_version, w, h,
              caps_.max_texture_size, caps_.max_ssbo_block_size);
  eglSwapInterval(display, 1);
  return true;
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

namespace rex::graphics::gl {
void* GlContext::CreateSharedContext() {
  EGLint attr[] = {EGL_CONTEXT_OPENGL_PROFILE_MASK_KHR, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT_KHR,
                   EGL_CONTEXT_MAJOR_VERSION_KHR,       4,
                   EGL_CONTEXT_MINOR_VERSION_KHR,       3,
                   EGL_NONE};
  EGLContext ctx = eglCreateContext(static_cast<EGLDisplay>(display_), static_cast<EGLConfig>(config_),
                                    static_cast<EGLContext>(context_), attr);
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
