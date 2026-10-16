/**
 * @file        graphics/gl/pipeline_cache.h
 * @brief       Shader translation (Xenos ucode -> SPIR-V -> GLSL) and linked GL program cache.
 *
 * Mesa nouveau (20.1) compiles + links one translated program in 0.5 .. 10+ s on the Switch CPU, and the GL
 * program binary path is unavailable (0 binary formats). So:
 *  - a compile thread translates AND (gl_link_thread, with a shared surfaceless EGL context) compiles and links
 *    programs; draws whose program is not ready yet are skipped instead of stalling the GPU thread;
 *  - every program the game uses is recorded on the SD card (guest ucode + translation modification, see
 *    gl_program_cache) and compiled in the background at the next boot ("warm-up"), so the menu / in-game
 *    programs are usually ready before they are first needed.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glad/glad.h>

#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/register_file.h>
#include <rex/string/buffer.h>

#include "shader.h"

namespace rex::graphics::gl {

struct GlProgram {
  GLuint program = 0;
  // kWorkerLinked: linked on the compile thread's shared context; usable once `fence` has signalled.
  enum State : int { kQueued = 0, kGlslReady, kLinking, kWorkerLinked, kReady, kFailed };
  std::atomic<int> state{kQueued};
  GLuint vs_id = 0, fs_id = 0;
  GLint sm_base_location = -1;  // xe_sm_base[4] uniform (shared memory window bases), -1 if unused
  GLsync fence = nullptr;  // null = handed over without a fence (see gl_link_fence)
  std::chrono::steady_clock::time_point link_start;
  std::chrono::steady_clock::time_point request_time;   // first draw that wanted it (urgent); guarded by mutex_
  std::chrono::steady_clock::time_point worker_linked;  // when the compile thread finished linking it
  GlShader::GlTranslation* vertex = nullptr;
  GlShader::GlTranslation* pixel = nullptr;
  // Guarded by GlPipelineCache::mutex_.
  bool queued = false;   // in urgent_ or warm_
  bool urgent = false;   // requested by a draw
  bool busy = false;     // being processed by the compile thread
  bool from_disk = false;
  size_t glsl_bytes = 0;  // VS + PS GLSL size once translated (link cost estimate for small-first scheduling)
  // GPU thread only.
  mutable bool used = false;  // a draw has used it (first-use draw accounting)
};

class GlPipelineCache {
 public:
  explicit GlPipelineCache(const RegisterFile& register_file) : register_file_(register_file) {}
  ~GlPipelineCache();

  bool Initialize(class GlContext* gl_context);
  void Shutdown();
  // Loads the persistent program list (gl_program_cache) and queues its programs for background compilation.
  // GPU thread, after Initialize.
  void WarmUpFromDisk();

  GlShader* LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                       uint32_t dword_count);
  void AnalyzeShaderUcode(Shader& shader) { shader.AnalyzeUcode(ucode_disasm_buffer_); }

  SpirvShaderTranslator::Modification GetCurrentVertexShaderModification(
      const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
      uint32_t interpolator_mask, bool ps_param_gen_used) const;
  SpirvShaderTranslator::Modification GetCurrentPixelShaderModification(
      const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
      reg::RB_DEPTHCONTROL normalized_depth_control) const;

  // Translates (if needed) the shader(s). ucode_buffer: scratch for ucode analysis (per thread).
  bool EnsureShadersTranslated(GlShader::GlTranslation* vertex, GlShader::GlTranslation* pixel,
                               string::StringBuffer& ucode_buffer);
  // Returns the linked program, or nullptr while it is being translated / compiled (or on failure).
  const GlProgram* GetProgram(GlShader::GlTranslation* vertex, GlShader::GlTranslation* pixel);

  uint32_t translated_shader_count() const { return translated_count_.load(); }
  uint32_t pending_programs() const { return pending_.load(); }
  // GPU thread, once per presented frame: refills the per-frame budget of deadline sync links.
  void OnFrame() { deadline_budget_ = 1; }
  bool links_on_worker() const { return link_on_worker_.load(); }

 private:
  const RegisterFile& register_file_;
  std::unique_ptr<SpirvShaderTranslator> translator_;
  string::StringBuffer ucode_disasm_buffer_;         // GPU thread
  std::unordered_map<uint64_t, std::unique_ptr<GlShader>> shaders_;

  struct ProgramKey {
    const void* vertex;
    const void* pixel;
    bool operator==(const ProgramKey& o) const { return vertex == o.vertex && pixel == o.pixel; }
  };
  struct ProgramKeyHash {
    size_t operator()(const ProgramKey& k) const {
      return std::hash<const void*>()(k.vertex) * 31 + std::hash<const void*>()(k.pixel);
    }
  };
  // Node-based: GlProgram addresses stay valid (the queues hold pointers).
  std::unordered_map<ProgramKey, GlProgram, ProgramKeyHash> programs_;
  void TranslateProgram(GlProgram& program, string::StringBuffer& ucode_buffer);
  void StartLink(GlProgram& program);
  // Returns true if linked (sets program.program / sm_base_location); false = failed (logged, state kFailed).
  bool FinishLink(GlProgram& program, bool block, bool* done = nullptr);
  void WorkerLink(GlProgram& program);
  struct Worker {
    GlPipelineCache* self = nullptr;
    int index = 0;
    void* thread = nullptr;   // libnx Thread*
    void* context = nullptr;  // shared EGL context (null: translate only)
    bool share_ok = false;    // its share probe passed: may link
    string::StringBuffer ucode_buffer;
  };
  void WorkerMain(Worker& worker);
  static void WorkerThunk(void* worker);
  void EnqueueLocked(GlProgram& program, bool urgent);
  bool WarmLinkAllowed();
  // GPU thread: stop linking on the compile thread (it only translates from now on); logs why.
  void FallBackToGpuThreadLinking(const char* why);
  // GPU thread, mutex_ held by `lock`: links a translated (or not yet translated) program on the GPU thread because a
  // draw has waited longer than gl_program_max_wait_ms for it. Returns the program if it is now ready.
  const GlProgram* DeadlineLinkLocked(GlProgram& p, std::unique_lock<std::mutex>& lock, double waited_ms);
  // GPU thread: checks the compile thread's progress (gl_link_thread_watchdog_ms) after a not-ready lookup.
  void CheckWorkerWatchdog();

  // Persistent program list.
  void RecordProgram(const GlProgram& program);
  void FlushCacheWrites();

  class GlContext* gl_context_ = nullptr;
  // gl_compile_threads compile threads, each with its own shared context (Mesa compiles one program per context at
  // a time; 0.5-10 s each on the Switch, and the first in-game frame needs dozens of them).
  std::vector<std::unique_ptr<Worker>> workers_;
  std::atomic<bool> link_on_worker_{false};
  int validate_left_ = 8;  // GPU thread: first worker-linked programs are checked on the GPU thread's context
  // Share-group probe: objects created on the GPU thread's context that the compile thread looks up on its shared
  // context at start-up (eglCreateContext's share_context is not guaranteed to be honoured by every EGL platform).
  GLuint probe_buffer_ = 0, probe_program_ = 0;
  GLsync probe_sync_ = nullptr;
  std::atomic<bool> fence_handover_{true};      // hand over with a GL fence (cleared if syncs are not shared)
  std::atomic<bool> worker_programs_usable_{false};  // share probe passed and no validation failure
  std::atomic<bool> worker_started_{false};     // the compile thread finished its start-up (context + probe)
  std::atomic<uint32_t> worker_links_done_{0};  // successful links on the compile thread
  std::atomic<int> worker_in_link_{0};          // compile threads inside glCompileShader / glLinkProgram
  std::atomic<uint32_t> worker_urgent_links_{0};  // ... of programs a draw had asked for
  std::mutex translate_mutex_;                  // translator_ + GlTranslation translation/GLSL (both threads)
  // GPU thread only.
  uint32_t handovers_ = 0;
  bool waiting_ = false;                        // a draw has been skipped since the last handover
  std::chrono::steady_clock::time_point first_wait_;
  int deadline_budget_ = 1;
  std::chrono::steady_clock::time_point last_deadline_link_;
  uint32_t deadline_links_ = 0;
  bool fence_problem_logged_ = false;
  bool first_link_seen_set_ = false;
  std::chrono::steady_clock::time_point first_link_seen_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<GlProgram*> urgent_, warm_;
  std::atomic<bool> running_{false};
  bool async_ = true;
  std::atomic<uint32_t> pending_{0};
  std::atomic<uint32_t> translated_count_{0};
  std::atomic<uint32_t> linked_count_{0};
  uint32_t dumped_ = 0;
  uint32_t dumped_big_ = 0;
  uint32_t warm_linked_ = 0, warm_skipped_ = 0;
  bool warm_logged_done_ = false;

  std::mutex cache_mutex_;
  bool cache_enabled_ = false;
  std::string cache_path_;
  std::vector<uint8_t> cache_pending_;
  std::unordered_set<uint64_t> cache_shaders_written_;
};

}  // namespace rex::graphics::gl
