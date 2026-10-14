/**
 * @file        graphics/gl/gl_context.h
 * @brief       EGL + OpenGL 4.3 core context for the Switch (libnx NWindow, Mesa/nouveau).
 */
#pragma once

#include <cstdint>
#include <string>

namespace rex::graphics::gl {

struct GlCaps {
  std::string renderer, version, glsl_version;
  int max_texture_size = 0;
  int max_ssbo_block_size = 0;
  int max_uniform_block_size = 0;
  int max_texture_units = 0;
  int max_color_attachments = 0;
  bool ext_texture_filter_anisotropic = false;
  bool arb_cull_distance = false;
  bool ext_demote_to_helper = false;
  bool s3tc = false;
};

// Single-context holder. All GL calls must happen on the thread that called Initialize().
class GlContext {
 public:
  ~GlContext() { Shutdown(); }
  bool Initialize();
  void Shutdown();
  void SwapBuffers();
  // Current default framebuffer size (re-queried; docked/handheld change).
  void QuerySurfaceSize(int& width, int& height) const;
  const GlCaps& caps() const { return caps_; }

 private:
  void* display_ = nullptr;
  void* surface_ = nullptr;
  void* context_ = nullptr;
  GlCaps caps_;
};

// Returns the GL error string for glGetError values.
const char* GlErrorString(unsigned int err);

}  // namespace rex::graphics::gl
