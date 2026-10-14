#include "command_processor.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <glad/glad.h>

#include <rex/cvar.h>
#include <rex/graphics/graphics_system.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/memory.h>

REXCVAR_DEFINE_STRING(gl_screenshot_dir, "", "GPU",
                      "GL backend: directory for presented-frame screenshots (.ppm); empty disables");
REXCVAR_DEFINE_INT32(gl_screenshot_swaps, 0, "GPU",
                     "GL backend: write a screenshot every N swaps (0 = only gl_screenshot_first)");
REXCVAR_DEFINE_INT32(gl_screenshot_first, 1, "GPU",
                     "GL backend: also screenshot the swap with this index (1-based, 0 = none)");

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
}  // namespace

struct GlCommandProcessor::Impl {
  GLuint blit_program = 0;
  GLuint vao = 0;
  GLuint front_tex = 0;
  uint32_t front_w = 0, front_h = 0;
  std::vector<uint32_t> front_pixels;
};

GlCommandProcessor::GlCommandProcessor(GraphicsSystem* graphics_system,
                                       system::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state), impl_(std::make_unique<Impl>()) {}

GlCommandProcessor::~GlCommandProcessor() = default;

bool GlCommandProcessor::SetupContext() {
  if (!context_.Initialize()) {
    return false;
  }
  glGenVertexArrays(1, &impl_->vao);
  GLuint vs = Compile(GL_VERTEX_SHADER, kBlitVs);
  GLuint fs = Compile(GL_FRAGMENT_SHADER, kBlitFs);
  impl_->blit_program = glCreateProgram();
  glAttachShader(impl_->blit_program, vs);
  glAttachShader(impl_->blit_program, fs);
  glLinkProgram(impl_->blit_program);
  glDeleteShader(vs);
  glDeleteShader(fs);
  glGenTextures(1, &impl_->front_tex);
  // Show something immediately so the window is not stale.
  glClearColor(0.f, 0.f, 0.25f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);
  context_.SwapBuffers();
  return true;
}

void GlCommandProcessor::ShutdownContext() {
  context_.Shutdown();
}

Shader* GlCommandProcessor::LoadShader(xenos::ShaderType, uint32_t, const uint32_t*, uint32_t) {
  return nullptr;
}

bool GlCommandProcessor::IssueDraw(xenos::PrimitiveType, uint32_t, IndexBufferInfo*, bool) {
  return true;
}

bool GlCommandProcessor::IssueCopy() {
  return true;
}

void GlCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                                   uint32_t frontbuffer_height) {
  ++swap_count_;
  // Decode the guest front buffer (fetch constant 0 is set by VdSwap) on the CPU: 32bpp, 32x32 tiled,
  // endian-swapped. This also serves as the debug view of what the guest wrote to memory.
  xenos::xe_gpu_texture_fetch_t fetch = register_file_->GetTextureFetch(0);
  uint32_t base = fetch.base_address << 12;
  uint32_t w = fetch.size_2d.width + 1, h = fetch.size_2d.height + 1;
  bool have = base != 0 && w > 1 && h > 1 && w <= 4096 && h <= 4096;
  if (swap_count_ <= 3 || (swap_count_ % 300) == 0) {
    REXGPU_INFO("GL swap #{}: ptr={:08X} {}x{} fetch base={:08X} {}x{} fmt={} endian={} tiled={} pitch={}",
                swap_count_, frontbuffer_ptr, frontbuffer_width, frontbuffer_height, base, w, h,
                uint32_t(fetch.format), uint32_t(fetch.endianness), uint32_t(fetch.tiled),
                uint32_t(fetch.pitch) << 5);
  }
  int sw = 1280, sh = 720;
  context_.QuerySurfaceSize(sw, sh);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, sw, sh);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glDisable(GL_SCISSOR_TEST);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  float pulse = float(swap_count_ & 63) / 63.0f;
  glClearColor(0.f, pulse * 0.2f, 0.2f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT);
  if (have) {
    uint32_t pitch = std::max<uint32_t>(uint32_t(fetch.pitch) << 5, w);
    const uint8_t* mem = memory_->TranslatePhysical(base);
    auto& px = impl_->front_pixels;
    px.resize(size_t(w) * h);
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t off = fetch.tiled ? uint32_t(texture_util::GetTiledOffset2D(int32_t(x), int32_t(y), pitch, 2))
                                   : (y * pitch + x) * 4;
        const uint8_t* b = mem + off;
        // Endian k8in32: bytes reversed within each dword -> R,G,B,A byte order.
        uint32_t v;
        switch (fetch.endianness) {
          case xenos::Endian::k8in32:
            v = uint32_t(b[3]) | uint32_t(b[2]) << 8 | uint32_t(b[1]) << 16 | uint32_t(b[0]) << 24;
            break;
          case xenos::Endian::k8in16:
            v = uint32_t(b[1]) | uint32_t(b[0]) << 8 | uint32_t(b[3]) << 16 | uint32_t(b[2]) << 24;
            break;
          case xenos::Endian::k16in32:
            v = uint32_t(b[2]) | uint32_t(b[3]) << 8 | uint32_t(b[0]) << 16 | uint32_t(b[1]) << 24;
            break;
          default:
            v = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
            break;
        }
        px[size_t(y) * w + x] = v;
      }
    }
    bool all_zero = std::all_of(px.begin(), px.end(), [](uint32_t v) { return v == 0; });
    if (all_zero) {
      have = false;  // guest memory untouched so far: keep the (animated) clear colour visible
    }
  }
  if (have) {
    auto& px = impl_->front_pixels;
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
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glUseProgram(impl_->blit_program);
    glBindVertexArray(impl_->vao);
    glActiveTexture(GL_TEXTURE0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
  }
  std::string dir = REXCVAR_GET(gl_screenshot_dir);
  if (!dir.empty()) {
    int every = REXCVAR_GET(gl_screenshot_swaps);
    int first = REXCVAR_GET(gl_screenshot_first);
    bool shot = (first && int(swap_count_) == first) || (every > 0 && (swap_count_ % every) == 0);
    if (shot) {
      WritePpm(dir + "/shot_" + std::to_string(swap_count_) + ".ppm", sw, sh);
    }
  }
  context_.SwapBuffers();
}

}  // namespace rex::graphics::gl
