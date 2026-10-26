/**
 * @file        graphics/gl/gl_shadow.h
 * @brief       Shadow of the GL bindings the draw path re-establishes on every draw (texture units, samplers,
 *              program, VAO, element array buffer, primitive restart). Mesa takes a mutex and does a hash lookup for
 *              every name-based bind call, which is a large part of the per-draw CPU cost on the Switch. The draw
 *              path calls the shadow::* helpers below (which skip redundant calls); every other call site goes
 *              through the macros at the end of this file (included by gl_ext.h), which invalidate the shadow first.
 *              GL thread only (the compile thread's context never calls the wrapped functions except
 *              glDeleteProgram, which only writes a constant).
 *              gl_state_shadow = false makes the helpers issue every call (the shadow is still maintained).
 */
#pragma once

#include <cstdint>

#include <glad/glad.h>

namespace rex::graphics::gl::shadow {

constexpr uint32_t kUnits = 32;
constexpr GLuint kUnknown = 0xFFFFFFFFu;

struct State {
  bool enabled = true;
  // Texture units.
  bool tex_known = false;
  GLuint tex[kUnits];
  GLenum tgt[kUnits];
  GLuint smp[kUnits];
  bool active_known = false;
  GLuint active = 0;
  // Draw state.
  GLuint program = kUnknown;
  GLuint vao = kUnknown;
  GLuint ebo = kUnknown;  // element array buffer bound to the draw VAO
  int restart = -1;
  // Diagnostics: redundant calls skipped.
  uint64_t skipped = 0, issued = 0;
};
inline State g;

inline void DirtyTex() {
  g.tex_known = false;
  g.active_known = false;
}
inline void DirtyProgram() { g.program = kUnknown; }
inline void DirtyVao() {
  g.vao = kUnknown;
  g.ebo = kUnknown;
}
inline void DirtyBuffers() { g.ebo = kUnknown; }
inline void DirtyRestart() { g.restart = -1; }
inline void DirtyAll() {
  DirtyTex();
  DirtyProgram();
  DirtyVao();
  DirtyRestart();
}

inline void ResetTex() {
  for (uint32_t i = 0; i < kUnits; ++i) {
    g.tex[i] = kUnknown;
    g.tgt[i] = 0;
    g.smp[i] = kUnknown;
  }
  g.tex_known = true;
}

inline void ActivateUnit(GLuint unit) {
  if (g.enabled && g.active_known && g.active == unit) {
    ++g.skipped;
    return;
  }
  ++g.issued;
  glad_glActiveTexture(GL_TEXTURE0 + unit);
  g.active = unit;
  g.active_known = true;
}

// Binds (target, tex) on the unit unless the shadow says it is already there.
inline void BindTexUnit(GLuint unit, GLenum target, GLuint tex) {
  if (!g.tex_known) ResetTex();
  if (g.enabled && g.tex[unit] == tex && g.tgt[unit] == target) {
    ++g.skipped;
    return;
  }
  ActivateUnit(unit);
  ++g.issued;
  glad_glBindTexture(target, tex);
  g.tex[unit] = tex;
  g.tgt[unit] = target;
}

inline void BindSamplerUnit(GLuint unit, GLuint sampler) {
  if (!g.tex_known) ResetTex();
  if (g.enabled && g.smp[unit] == sampler) {
    ++g.skipped;
    return;
  }
  ++g.issued;
  glad_glBindSampler(unit, sampler);
  g.smp[unit] = sampler;
}

inline void UseProgram(GLuint program) {
  if (g.enabled && g.program == program) {
    ++g.skipped;
    return;
  }
  ++g.issued;
  glad_glUseProgram(program);
  g.program = program;
}

inline void BindVertexArray(GLuint vao) {
  if (g.enabled && g.vao == vao) {
    ++g.skipped;
    return;
  }
  ++g.issued;
  glad_glBindVertexArray(vao);
  g.vao = vao;
  g.ebo = kUnknown;  // the element array binding is per VAO: not tracked across VAO changes
}

// Element array buffer of the currently bound (draw) VAO.
inline void BindElementArray(GLuint buffer) {
  if (g.enabled && g.ebo == buffer) {
    ++g.skipped;
    return;
  }
  ++g.issued;
  glad_glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
  g.ebo = buffer;
}

inline void PrimitiveRestartFixed(bool on) {
  if (g.enabled && g.restart == int(on)) {
    ++g.skipped;
    return;
  }
  ++g.issued;
  if (on) {
    glad_glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
  } else {
    glad_glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
  }
  g.restart = int(on);
}

}  // namespace rex::graphics::gl::shadow

// Every call site that is not the draw path's own tracking invalidates the shadow (the helpers above use the glad_*
// pointers directly, so they are not affected).
#define glActiveTexture(...) (::rex::graphics::gl::shadow::DirtyTex(), glad_glActiveTexture(__VA_ARGS__))
#define glBindTexture(...) (::rex::graphics::gl::shadow::DirtyTex(), glad_glBindTexture(__VA_ARGS__))
#define glBindSampler(...) (::rex::graphics::gl::shadow::DirtyTex(), glad_glBindSampler(__VA_ARGS__))
#define glDeleteTextures(...) (::rex::graphics::gl::shadow::DirtyTex(), glad_glDeleteTextures(__VA_ARGS__))
#define glDeleteSamplers(...) (::rex::graphics::gl::shadow::DirtyTex(), glad_glDeleteSamplers(__VA_ARGS__))
#define glUseProgram(...) (::rex::graphics::gl::shadow::DirtyProgram(), glad_glUseProgram(__VA_ARGS__))
#define glDeleteProgram(...) (::rex::graphics::gl::shadow::DirtyProgram(), glad_glDeleteProgram(__VA_ARGS__))
#define glBindVertexArray(...) (::rex::graphics::gl::shadow::DirtyVao(), glad_glBindVertexArray(__VA_ARGS__))
#define glDeleteVertexArrays(...) (::rex::graphics::gl::shadow::DirtyVao(), glad_glDeleteVertexArrays(__VA_ARGS__))
#define glDeleteBuffers(...) (::rex::graphics::gl::shadow::DirtyBuffers(), glad_glDeleteBuffers(__VA_ARGS__))
