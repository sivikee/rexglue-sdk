#include "pipeline_cache.h"
#include "perf.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include <malloc.h>
#include <unistd.h>

#include <switch.h>
#include <xxhash.h>
#include "gl_context.h"
#include "gl_ext.h"

#include <rex/cvar.h>
#include <rex/graphics/util/draw.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(gl_parallel_compile, false, "GPU", "GL backend: enable GL_KHR_parallel_shader_compile driver threads");
REXCVAR_DEFINE_BOOL(gl_async_compile, true, "GPU", "GL backend: translate and compile shader programs on a worker thread (draws are skipped until ready)");
REXCVAR_DEFINE_BOOL(gl_link_thread, true, "GPU",
                    "GL backend: the compile thread also compiles + links GL programs on a shared surfaceless EGL context "
                    "(Mesa takes 0.5-10 s per program on the Switch; false = link on the GPU thread, which stalls the frame)");
REXCVAR_DEFINE_INT32(gl_compile_thread_priority, 0x2C, "GPU",
                     "GL backend: Horizon priority of the compile thread for programs a draw is waiting for (0x2C = like the GPU thread)");
REXCVAR_DEFINE_INT32(gl_warm_thread_priority, 0x2E, "GPU",
                     "GL backend: compile thread priority while it pre-compiles programs from the program cache (higher = lower priority)");
REXCVAR_DEFINE_BOOL(gl_program_cache, true, "GPU",
                    "GL backend: record every program the game uses (guest ucode + modification) in gl_program_cache_path "
                    "and pre-compile them in the background at the next start");
REXCVAR_DEFINE_STRING(gl_program_cache_path, "sdmc:/oblivion/cache/gl_programs.bin", "GPU",
                      "GL backend: program cache file (delete it to start over)");
REXCVAR_DEFINE_INT32(gl_program_warm_max, 800, "GPU", "GL backend: at most this many cached programs are pre-compiled per start (0 = none)");
REXCVAR_DEFINE_INT32(gl_warm_min_headroom_mb, 512, "GPU",
                     "GL backend: background pre-compilation pauses while less than this much malloc memory is left");
REXCVAR_DEFINE_STRING(gl_dump_shaders, "", "GPU",
                      "GL backend: directory to dump generated GLSL of failing/first shaders (empty = off)");

REXCVAR_DEFINE_INT32(gl_debug_ps_strip, 0, "GPU", "GL debug (gl_debug_ps=5): 1 = drop SSBO decl, 2 = drop UBO decls, 4 = drop interpolator input");
REXCVAR_DEFINE_INT32(gl_debug_vs, 0, "GPU", "GL debug: 1 = vertex shaders output a fullscreen triangle position");
REXCVAR_DEFINE_INT32(gl_debug_ps, 0, "GPU",
                     "GL backend debug: 1 = pixel shaders output magenta, 2 = output interpolator 0, 3 = output yellow if system constants flags look sane");

extern "C" char* fake_heap_end;  // libnx: end of the newlib malloc region

namespace rex::graphics::gl {

namespace {
// Program cache file format. Bump kCacheGenTag whenever the translator, its features (Initialize) or the GLSL
// post-processing (spirv_to_glsl.cpp) change in a way that changes the translation of a (ucode, modification).
constexpr char kCacheMagic[8] = {'R', 'X', 'G', 'L', 'P', 'C', '0', '1'};
constexpr uint32_t kCacheFormat = 1;
constexpr uint32_t kCacheGenTag = 1;
constexpr uint8_t kRecShader = 'S', kRecProgram = 'P';

uint64_t MallocHeadroom() {
  struct mallinfo mi = mallinfo();
  char* brk = static_cast<char*>(sbrk(0));
  if (!fake_heap_end || !brk || brk == reinterpret_cast<char*>(-1) || brk > fake_heap_end) return uint64_t(mi.fordblks);
  return uint64_t(fake_heap_end - brk) + uint64_t(mi.fordblks);
}

template <typename T>
void Put(std::vector<uint8_t>& out, const T& v) {
  const size_t n = out.size();
  out.resize(n + sizeof(T));
  std::memcpy(out.data() + n, &v, sizeof(T));
}
}  // namespace

GlPipelineCache::~GlPipelineCache() {
  Shutdown();
}

bool GlPipelineCache::Initialize(GlContext* gl_context) {
  gl_context_ = gl_context;
  SpirvShaderTranslator::Features features(false);
  features.max_storage_buffer_range = 128 * 1024 * 1024;
  features.full_draw_index_uint32 = true;
  features.vertex_pipeline_stores_and_atomics = false;  // memexport not supported yet
  features.fragment_stores_and_atomics = false;
  features.clip_distance = true;
  features.cull_distance = true;
  features.image_view_format_swizzle = false;  // host swizzle done in the shader via system constants
  features.signed_zero_inf_nan_preserve_float32 = false;
  features.denorm_flush_to_zero_float32 = false;
  features.rounding_mode_rte_float32 = false;
  features.fragment_shader_sample_interlock = false;
  features.demote_to_helper_invocation = false;
  features.sample_rate_shading = false;
  translator_ = std::make_unique<SpirvShaderTranslator>(features, false, false, false, 1, 1);
  if (glMaxShaderCompilerThreadsKHR_ && REXCVAR_GET(gl_parallel_compile)) {
    glMaxShaderCompilerThreadsKHR_(0xFFFFFFFFu);
  }
  {
    GLint formats = -1;
    glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &formats);
    while (glGetError() != GL_NO_ERROR) {}
    REXGPU_INFO("GL: GL_NUM_PROGRAM_BINARY_FORMATS = {} (program binaries {})", formats,
                formats > 0 ? "available (not used yet)" : "unsupported by this Mesa/nouveau: the program cache stores guest ucode and recompiles");
  }
  async_ = REXCVAR_GET(gl_async_compile);
  if (async_) {
    if (REXCVAR_GET(gl_link_thread)) {
      // Created here (GPU thread, main context current), made current on the compile thread.
      worker_context_ = gl_context_->CreateSharedContext();
      if (!worker_context_) {
        REXGPU_WARN("GL: no shared context for the compile thread: programs are linked on the GPU thread");
      }
    }
    running_ = true;
    Thread* t = new Thread();
    // 8 MiB stack: Mesa's GLSL front end recurses deeply on the translator's big shaders.
    if (R_FAILED(threadCreate(t, &GlPipelineCache::WorkerThunk, this, nullptr, 0x800000, 0x2C, -2)) ||
        R_FAILED(threadStart(t))) {
      REXGPU_ERROR("GL: failed to start the shader compile thread");
      running_ = false;
      async_ = false;
      delete t;
    } else {
      worker_thread_ = t;
    }
  }
  return true;
}

void GlPipelineCache::Shutdown() {
  if (worker_thread_) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
    }
    cv_.notify_all();
    threadWaitForExit(static_cast<Thread*>(worker_thread_));
    threadClose(static_cast<Thread*>(worker_thread_));
    delete static_cast<Thread*>(worker_thread_);
    worker_thread_ = nullptr;
  }
  FlushCacheWrites();
  for (auto& p : programs_) {
    if (p.second.program) {
      glDeleteProgram(p.second.program);
    }
  }
  programs_.clear();
  shaders_.clear();
}

GlShader* GlPipelineCache::LoadShader(xenos::ShaderType shader_type, const uint32_t* host_address,
                                      uint32_t dword_count) {
  uint64_t hash = XXH3_64bits(host_address, dword_count * sizeof(uint32_t));
  auto it = shaders_.find(hash);
  if (it != shaders_.end()) {
    return it->second.get();
  }
  {
    static int logged = 0;
    if (logged++ < 40) {
      REXGPU_INFO("GL LoadShader type={} dwords={} hash={:016X} data={:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                  uint32_t(shader_type), dword_count, hash, dword_count > 0 ? host_address[0] : 0,
                  dword_count > 1 ? host_address[1] : 0, dword_count > 2 ? host_address[2] : 0,
                  dword_count > 3 ? host_address[3] : 0, dword_count > 4 ? host_address[4] : 0,
                  dword_count > 5 ? host_address[5] : 0);
    }
  }
  auto shader = std::make_unique<GlShader>(shader_type, hash, host_address, dword_count);
  GlShader* ptr = shader.get();
  shaders_.emplace(hash, std::move(shader));
  return ptr;
}

SpirvShaderTranslator::Modification GlPipelineCache::GetCurrentVertexShaderModification(
    const Shader& shader, Shader::HostVertexShaderType host_vertex_shader_type,
    uint32_t interpolator_mask, bool ps_param_gen_used) const {
  const auto& regs = register_file_;
  SpirvShaderTranslator::Modification modification(translator_->GetDefaultVertexShaderModification(
      shader.GetDynamicAddressableRegisterCount(regs.Get<reg::SQ_PROGRAM_CNTL>().vs_num_reg),
      host_vertex_shader_type));
  modification.vertex.interpolator_mask = interpolator_mask;
  if (Shader::IsHostVertexShaderTypeDomain(host_vertex_shader_type)) {
    modification.vertex.tessellation_mode = uint32_t(regs.Get<reg::VGT_HOS_CNTL>().tess_mode);
  } else {
    modification.vertex.tessellation_mode = 0;
  }
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  uint32_t user_clip_planes = pa_cl_clip_cntl.clip_disable ? 0 : pa_cl_clip_cntl.ucp_ena;
  modification.vertex.user_clip_plane_cull =
      (user_clip_planes && pa_cl_clip_cntl.ucp_cull_only_ena) ? 1 : 0;
  modification.vertex.user_clip_plane_count = rex::bit_count(user_clip_planes);
  modification.vertex.point_ps_ucp_mode = pa_cl_clip_cntl.ps_ucp_mode;
  if (host_vertex_shader_type == Shader::HostVertexShaderType::kPointListAsTriangleStrip) {
    modification.vertex.output_point_parameters = uint32_t(ps_param_gen_used);
  } else {
    modification.vertex.output_point_parameters =
        uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b001) &&
                 regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type == xenos::PrimitiveType::kPointList);
  }
  modification.vertex.vertex_kill_and = uint32_t((shader.writes_point_size_edge_flag_kill_vertex() & 0b100) &&
                                                 !pa_cl_clip_cntl.vtx_kill_or);
  return modification;
}

SpirvShaderTranslator::Modification GlPipelineCache::GetCurrentPixelShaderModification(
    const Shader& shader, uint32_t interpolator_mask, uint32_t param_gen_pos,
    reg::RB_DEPTHCONTROL normalized_depth_control) const {
  const auto& regs = register_file_;
  SpirvShaderTranslator::Modification modification(translator_->GetDefaultPixelShaderModification(
      shader.GetDynamicAddressableRegisterCount(regs.Get<reg::SQ_PROGRAM_CNTL>().ps_num_reg)));
  modification.pixel.interpolator_mask = interpolator_mask;
  modification.pixel.interpolators_centroid =
      interpolator_mask & ~xenos::GetInterpolatorSamplingPattern(
                              regs.Get<reg::RB_SURFACE_INFO>().msaa_samples,
                              regs.Get<reg::SQ_CONTEXT_MISC>().sc_sample_cntl,
                              regs.Get<reg::SQ_INTERPOLATOR_CNTL>().sampling_pattern);
  if (param_gen_pos < xenos::kMaxInterpolators) {
    modification.pixel.param_gen_enable = 1;
    modification.pixel.param_gen_interpolator = param_gen_pos;
    modification.pixel.param_gen_point =
        uint32_t(regs.Get<reg::VGT_DRAW_INITIATOR>().prim_type == xenos::PrimitiveType::kPointList);
  } else {
    modification.pixel.param_gen_enable = 0;
    modification.pixel.param_gen_interpolator = 0;
    modification.pixel.param_gen_point = 0;
  }
  using DepthStencilMode = SpirvShaderTranslator::Modification::DepthStencilMode;
  if (shader.implicit_early_z_write_allowed() &&
      (!shader.writes_color_target(0) ||
       !draw_util::DoesCoverageDependOnAlpha(regs.Get<reg::RB_COLORCONTROL>()))) {
    modification.pixel.depth_stencil_mode = DepthStencilMode::kEarlyHint;
  } else {
    modification.pixel.depth_stencil_mode = DepthStencilMode::kNoModifiers;
  }
  return modification;
}

bool GlPipelineCache::EnsureShadersTranslated(GlShader::GlTranslation* vertex,
                                              GlShader::GlTranslation* pixel,
                                              string::StringBuffer& ucode_buffer) {
  auto translate = [&](GlShader::GlTranslation* t) -> bool {
    if (!t->is_translated()) {
      t->shader().AnalyzeUcode(ucode_buffer);
      auto t0 = std::chrono::steady_clock::now();
      if (!translator_->TranslateAnalyzedShader(*t)) {
        REXGPU_ERROR("GL: shader {:016X} translation failed", t->shader().ucode_data_hash());
        return false;
      }
      ++translated_count_;
      double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (translated_count_ <= 8 || translated_count_ % 25 == 0) {
        REXGPU_INFO("GL: translated shader #{} {:016X} mod {:016X} ({}) in {:.1f} ms, SPIR-V {} bytes",
                    translated_count_.load(), t->shader().ucode_data_hash(), t->modification(),
                    t->shader().type() == xenos::ShaderType::kVertex ? "VS" : "PS", ms,
                    t->translated_binary().size());
      }
    }
    return t->is_valid();
  };
  if (!translate(vertex)) {
    return false;
  }
  if (pixel && !translate(pixel)) {
    return false;
  }
  return true;
}

// Caller holds mutex_.
void GlPipelineCache::EnqueueLocked(GlProgram& program, bool urgent) {
  if (urgent && !program.urgent) {
    program.urgent = true;
    ++pending_;
    if (program.queued) {
      // Promote a queued warm-up entry.
      for (auto it = warm_.begin(); it != warm_.end(); ++it) {
        if (*it == &program) {
          warm_.erase(it);
          break;
        }
      }
      program.queued = false;
    }
  }
  if (program.queued || program.busy) {
    return;
  }
  program.queued = true;
  (program.urgent ? urgent_ : warm_).push_back(&program);
  perf::SetWorkerQueue(uint32_t(urgent_.size()), uint32_t(warm_.size()));
  cv_.notify_one();
}

const GlProgram* GlPipelineCache::GetProgram(GlShader::GlTranslation* vertex,
                                             GlShader::GlTranslation* pixel) {
  perf::Activity act(perf::kPhProgramLookup);
  ProgramKey key{vertex, pixel};
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = programs_.find(key);
  if (it == programs_.end()) {
    GlProgram& program = programs_[key];
    program.vertex = vertex;
    program.pixel = pixel;
    RecordProgram(program);
    if (async_) {
      EnqueueLocked(program, true);
      return nullptr;
    }
    lock.unlock();
    TranslateProgram(program, ucode_disasm_buffer_);
    if (program.state.load() == GlProgram::kGlslReady) {
      perf::Activity link_act(perf::kPhLinkSync, uint32_t(vertex->shader().ucode_data_hash()),
                              pixel ? uint32_t(pixel->shader().ucode_data_hash()) : 0);
      perf::Scope link_scope(perf::kLinkTicks);
      StartLink(program);
      if (FinishLink(program, true)) {
        program.state = GlProgram::kReady;
        perf::Add(perf::kLinkCount);
      }
    }
    FlushCacheWrites();
    return program.state.load() == GlProgram::kReady ? &program : nullptr;
  }
  GlProgram& p = it->second;
  if (p.busy) {
    return nullptr;  // the compile thread owns it right now (translating / linking)
  }
  switch (p.state.load()) {
    case GlProgram::kQueued:
      if (async_) EnqueueLocked(p, true);
      return nullptr;
    case GlProgram::kGlslReady:
      if (async_ && link_on_worker_.load()) {
        EnqueueLocked(p, true);
        return nullptr;
      }
      lock.unlock();
      {
        // Synchronous link on the GPU thread (gl_link_thread = false or no shared context): stalls the frame.
        perf::Activity link_act(perf::kPhLinkSync, uint32_t(p.vertex->shader().ucode_data_hash()),
                                p.pixel ? uint32_t(p.pixel->shader().ucode_data_hash()) : 0);
        perf::Scope link_scope(perf::kLinkTicks);
        StartLink(p);
      }
      return nullptr;
    case GlProgram::kLinking: {
      lock.unlock();
      perf::Scope link_scope(perf::kLinkTicks);
      bool done = false;
      if (FinishLink(p, false, &done)) {
        p.state = GlProgram::kReady;
        perf::Add(perf::kLinkCount);
      }
      (void)done;
      return p.state.load() == GlProgram::kReady ? &p : nullptr;
    }
    case GlProgram::kWorkerLinked: {
      lock.unlock();
      if (!link_on_worker_.load()) {
        // Worker programs were found unusable on this context: relink here.
        p.program = 0;
        p.state = GlProgram::kGlslReady;
        return nullptr;
      }
      GLenum r = glClientWaitSync(p.fence, 0, 0);
      if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) {
        return nullptr;
      }
      glDeleteSync(p.fence);
      p.fence = nullptr;
      if (validate_left_ > 0) {
        --validate_left_;
        GLint linked = 0;
        const bool is_program = glIsProgram(p.program) == GL_TRUE;
        if (is_program) glGetProgramiv(p.program, GL_LINK_STATUS, &linked);
        if (!is_program || !linked) {
          REXGPU_ERROR(
              "GL: program {} linked on the compile thread is not usable on the GPU thread (glIsProgram={} link={}); "
              "linking on the GPU thread from now on",
              p.program, is_program, linked);
          link_on_worker_ = false;
          p.program = 0;
          p.state = GlProgram::kGlslReady;
          return nullptr;
        }
        if (validate_left_ == 7) {
          REXGPU_INFO("GL: first program linked on the compile thread is valid on the GPU thread (program {})",
                      p.program);
        }
      }
      p.state = GlProgram::kReady;
      return &p;
    }
    case GlProgram::kReady:
      return &p;
    default:
      return nullptr;
  }
}

void GlPipelineCache::WorkerThunk(void* self) {
  static_cast<GlPipelineCache*>(self)->WorkerMain();
}

bool GlPipelineCache::WarmLinkAllowed() {
  const uint64_t headroom = MallocHeadroom();
  const uint64_t need = uint64_t(std::max(0, REXCVAR_GET(gl_warm_min_headroom_mb))) << 20;
  return headroom >= need;
}

// Compile thread: Xenos ucode -> SPIR-V -> GLSL, and with gl_link_thread also glCompileShader / glLinkProgram on
// a shared context. Programs a draw waits for (urgent_) go first; cached ones (warm_) run at a lower priority.
void GlPipelineCache::WorkerMain() {
  svcSetThreadCoreMask(CUR_THREAD_HANDLE, -1, 0x7);
  if (worker_context_) {
    if (gl_context_->MakeCurrentWorker(worker_context_)) {
      link_on_worker_ = true;
      REXGPU_INFO("GL: compile thread has a shared context: programs are compiled and linked off the GPU thread");
    } else {
      REXGPU_WARN("GL: compile thread could not make the shared context current: programs are linked on the GPU thread");
    }
  }
  int cur_prio = -1;
  while (true) {
    GlProgram* program = nullptr;
    bool urgent = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !running_ || !urgent_.empty() || !warm_.empty(); });
      if (!running_) {
        break;
      }
      if (!urgent_.empty()) {
        program = urgent_.front();
        urgent_.pop_front();
      } else {
        program = warm_.front();
        warm_.pop_front();
      }
      program->queued = false;
      program->busy = true;
      urgent = program->urgent;
      perf::SetWorkerQueue(uint32_t(urgent_.size()), uint32_t(warm_.size()));
    }
    const int prio = urgent ? REXCVAR_GET(gl_compile_thread_priority) : REXCVAR_GET(gl_warm_thread_priority);
    if (prio != cur_prio && prio >= 0x18 && prio <= 0x3F) {
      svcSetThreadPriority(CUR_THREAD_HANDLE, prio);
      cur_prio = prio;
    }
    const uint64_t vh = program->vertex->shader().ucode_data_hash();
    const uint64_t ph = program->pixel ? program->pixel->shader().ucode_data_hash() : 0;
    if (program->state.load() == GlProgram::kQueued) {
      perf::SetWorkerActivity(perf::kWTranslate, vh, ph);
      TranslateProgram(*program, worker_ucode_disasm_buffer_);
    }
    bool skipped_warm = false;
    if (link_on_worker_.load() && program->state.load() == GlProgram::kGlslReady) {
      if (!urgent && !WarmLinkAllowed()) {
        skipped_warm = true;  // left translated; linked when a draw asks for it
      } else {
        perf::SetWorkerActivity(perf::kWCompileLink, vh, ph);
        WorkerLink(*program);
      }
    }
    perf::SetWorkerActivity(perf::kWIdle);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      program->busy = false;
      const bool needs_link = program->state.load() == GlProgram::kGlslReady && link_on_worker_.load();
      if (program->urgent) {
        if (needs_link) {
          // A draw started waiting while this was a warm job whose link was skipped: link it now.
          if (!program->queued) {
            program->queued = true;
            urgent_.push_back(program);
          }
        } else {
          --pending_;
        }
      } else if (skipped_warm) {
        ++warm_skipped_;
      } else {
        ++warm_linked_;
      }
      (void)urgent;
      if (!warm_logged_done_ && warm_.empty() && (warm_linked_ || warm_skipped_)) {
        warm_logged_done_ = true;
        REXGPU_INFO("GL program cache: background pre-compilation done ({} linked, {} skipped for low memory; malloc headroom {} MiB)",
                    warm_linked_, warm_skipped_, MallocHeadroom() >> 20);
      }
    }
    FlushCacheWrites();
  }
  if (worker_context_) {
    gl_context_->ReleaseCurrentWorker();
  }
}

void GlPipelineCache::WorkerLink(GlProgram& program) {
  perf::Scope link_scope(perf::kWLinkTicks);
  StartLink(program);
  if (!FinishLink(program, true)) {
    return;  // kFailed, logged
  }
  program.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glFlush();
  perf::Add(perf::kWLinkCount);
  program.state = GlProgram::kWorkerLinked;
}

void GlPipelineCache::TranslateProgram(GlProgram& program, string::StringBuffer& ucode_buffer) {
  perf::Add(perf::kXlateCount);
  perf::Scope perf_xlate_scope(perf::kXlateTicks);
  GlShader::GlTranslation* vertex = program.vertex;
  GlShader::GlTranslation* pixel = program.pixel;
  if (!EnsureShadersTranslated(vertex, pixel, ucode_buffer)) {
    program.state = GlProgram::kFailed;
    return;
  }
  const GlslShader* vs_glsl = vertex->GetOrCreateGlsl();
  const GlslShader* ps_glsl = pixel ? pixel->GetOrCreateGlsl() : nullptr;
  if (!vs_glsl || (pixel && !ps_glsl)) {
    program.state = GlProgram::kFailed;
    return;
  }
  program.state = GlProgram::kGlslReady;
}

void GlPipelineCache::StartLink(GlProgram& program) {
  const GlslShader* vs_glsl = program.vertex->glsl();
  const GlslShader* ps_glsl = program.pixel ? program.pixel->glsl() : nullptr;
  program.link_start = std::chrono::steady_clock::now();
  auto make = [&](GLenum type, const std::string& source) {
    GLuint s = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);  // asynchronous with GL_KHR_parallel_shader_compile
    return s;
  };
  std::string vs_source = vs_glsl->source;
  if (REXCVAR_GET(gl_debug_vs) == 2) {
    vs_source = "#version 430\nout gl_PerVertex { vec4 gl_Position; };\nvoid main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";
  } else if (REXCVAR_GET(gl_debug_vs) == 1) {
    size_t m = vs_source.find("void main()");
    if (m != std::string::npos) {
      vs_source.replace(m, 11, "void xe_vs_orig()");
      vs_source += "\nvoid main() { xe_vs_orig(); vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";
    }
  }
  if (REXCVAR_GET(gl_debug_vs) >= 3) {
    size_t m = vs_source.find("void main()");
    if (m != std::string::npos) {
      vs_source.erase(m);
      int strip = REXCVAR_GET(gl_debug_ps_strip);
      auto erase_block = [&](const char* marker) {
        size_t mk = vs_source.find(marker);
        while (mk != std::string::npos) {
          size_t st = vs_source.rfind("layout(", mk);
          size_t en = vs_source.find(";", mk);
          if (st == std::string::npos || en == std::string::npos) break;
          vs_source.erase(st, en - st + 1);
          mk = vs_source.find(marker);
        }
      };
      if (strip & 1) erase_block("xe_shared_memory[4]");
      if (strip & 2) { erase_block("xe_uniform_system_constants;"); erase_block("xe_uniform_bool_loop_constants;"); erase_block("xe_uniform_fetch_constants;"); erase_block("xe_uniform_float_constants;"); }
      const char* extra = "";
      if (REXCVAR_GET(gl_debug_vs) == 4) extra = " if (xe_sm_fetch(uint(gl_VertexID) & 15u) == 0xDEADBEEFu) gl_Position.z = 0.5;";
      if (REXCVAR_GET(gl_debug_vs) == 5) extra = " if (xe_uniform_system_constants.flags == 0xFFFFFFFFu) gl_Position.z = 0.5;";
      if (REXCVAR_GET(gl_debug_vs) == 6) extra = " if (xe_uniform_fetch_constants.fetch_constants[uint(gl_VertexID) & 7u].x == 0xDEADBEEFu) gl_Position.z = 0.5;";
      vs_source += "void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);"; vs_source += extra; vs_source += " }\n";
    }
  }
  program.vs_id = make(GL_VERTEX_SHADER, vs_source);
  std::string ps_source = ps_glsl ? ps_glsl->source : std::string();
  if (ps_glsl && REXCVAR_GET(gl_debug_ps) == 9) {
    ps_source = "#version 430\nlayout(location = 0) out vec4 xe_out_fragment_data_0;\nvoid main() { xe_out_fragment_data_0 = vec4(1.0, 0.0, 1.0, 1.0); }\n";
  } else if (ps_glsl && REXCVAR_GET(gl_debug_ps) > 0 && REXCVAR_GET(gl_debug_ps) <= 2 && ps_source.find("xe_out_fragment_data_0") != std::string::npos) {
    size_t m = ps_source.find("void main()");
    if (m != std::string::npos) {
      ps_source.replace(m, 11, "void xe_main_orig()");
      int mode = REXCVAR_GET(gl_debug_ps);
      if (mode == 2 && ps_source.find("xe_in_interpolator_0") != std::string::npos) {
        ps_source += "\nvoid main() { xe_main_orig(); xe_out_fragment_data_0 = vec4(xe_in_interpolator_0.xyz, 1.0); }\n";
      } else {
        ps_source += "\nvoid main() { xe_main_orig(); xe_out_fragment_data_0 = vec4(1.0, 0.0, 1.0, 1.0); }\n";
      }
    }
  }
  if (ps_glsl && (REXCVAR_GET(gl_debug_ps) == 3 || REXCVAR_GET(gl_debug_ps) == 4)) {
    auto strip = [&](const char* what) {
      size_t p;
      while ((p = ps_source.find(what)) != std::string::npos) ps_source.erase(p, std::strlen(what));
    };
    strip("layout(early_fragment_tests) in;");
    strip("invariant ");
    strip("precise ");
    if (REXCVAR_GET(gl_debug_ps) == 4) {
      size_t m = ps_source.find("void main()");
      if (m != std::string::npos) {
        ps_source.replace(m, 11, "void xe_main_orig()");
        ps_source += "\nvoid main() { xe_main_orig(); xe_out_fragment_data_0 = vec4(1.0, 0.0, 1.0, 1.0); }\n";
      }
    }
  }
  if (ps_glsl && REXCVAR_GET(gl_debug_ps) == 5) {
    size_t m = ps_source.find("void main()");
    if (m != std::string::npos) {
      ps_source.erase(m);
      int strip = REXCVAR_GET(gl_debug_ps_strip);
      auto erase_block = [&](const char* marker) {
        size_t mk = ps_source.find(marker);
        while (mk != std::string::npos) {
          size_t st = ps_source.rfind("layout(", mk);
          size_t en = ps_source.find(";", mk);
          if (st == std::string::npos || en == std::string::npos) break;
          ps_source.erase(st, en - st + 1);
          mk = ps_source.find(marker);
        }
      };
      if (strip & 1) erase_block("xe_shared_memory[4]");
      if (strip & 2) { erase_block("xe_uniform_system_constants;"); erase_block("xe_uniform_bool_loop_constants;"); erase_block("xe_uniform_fetch_constants;"); erase_block("xe_uniform_float_constants;"); }
      if (strip & 4) erase_block("xe_in_interpolator_0;");
      if (strip & 8) { size_t p = ps_source.find("layout(early_fragment_tests) in;"); if (p != std::string::npos) ps_source.erase(p, 32); p = ps_source.find("invariant "); if (p != std::string::npos) ps_source.erase(p, 10); }
      ps_source += "void main() { xe_out_fragment_data_0 = vec4(1.0, 0.0, 1.0, 1.0); }\n";
    }
  }
  program.fs_id = ps_glsl ? make(GL_FRAGMENT_SHADER, ps_source) : 0;
  program.program = glCreateProgram();
  glAttachShader(program.program, program.vs_id);
  if (program.fs_id) {
    glAttachShader(program.program, program.fs_id);
  }
  glLinkProgram(program.program);
  program.state = GlProgram::kLinking;
}

bool GlPipelineCache::FinishLink(GlProgram& program, bool block, bool* done_out) {
  if (done_out) *done_out = false;
  if (!block) {
    GLint done = 1;
    if (glMaxShaderCompilerThreadsKHR_ && REXCVAR_GET(gl_parallel_compile)) {
      glGetProgramiv(program.program, 0x91B1 /* GL_COMPLETION_STATUS_KHR */, &done);
    }
    if (!done) {
      return false;
    }
  }
  if (done_out) *done_out = true;
  auto dump = [&](const char* tag, const std::string& source, uint64_t hash) {
    std::string dir = REXCVAR_GET(gl_dump_shaders);
    if (dir.empty()) {
      return;
    }
    char name[256];
    std::snprintf(name, sizeof(name), "%s/%s_%016llX.glsl", dir.c_str(), tag,
                  (unsigned long long)hash);
    if (FILE* f = std::fopen(name, "wb")) {
      std::fwrite(source.data(), 1, source.size(), f);
      std::fclose(f);
    }
  };
  GlShader::GlTranslation* vertex = program.vertex;
  GlShader::GlTranslation* pixel = program.pixel;
  GLint ok = 0;
  glGetProgramiv(program.program, GL_LINK_STATUS, &ok);
  std::string log;
  bool result = false;
  if (!ok) {
    auto shader_log = [&](GLuint s, const char* what) {
      GLint cs = 0;
      glGetShaderiv(s, GL_COMPILE_STATUS, &cs);
      if (!cs) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        log.assign(size_t(len > 1 ? len : 1), 0);
        glGetShaderInfoLog(s, len, nullptr, log.data());
        REXGPU_ERROR("GL: {} shader failed to compile: {}", what, log);
      }
    };
    shader_log(program.vs_id, "vertex");
    if (program.fs_id) {
      shader_log(program.fs_id, "pixel");
    }
    GLint len = 0;
    glGetProgramiv(program.program, GL_INFO_LOG_LENGTH, &len);
    log.assign(size_t(len > 1 ? len : 1), 0);
    glGetProgramInfoLog(program.program, len, nullptr, log.data());
    REXGPU_ERROR("GL: program link failed (vs {:016X}, ps {:016X}): {}",
                 vertex->shader().ucode_data_hash(),
                 pixel ? pixel->shader().ucode_data_hash() : uint64_t(0), log);
    dump("vs_fail", vertex->glsl()->source, vertex->shader().ucode_data_hash());
    if (pixel) {
      dump("ps_fail", pixel->glsl()->source, pixel->shader().ucode_data_hash());
    }
    glDeleteProgram(program.program);
    program.program = 0;
    program.state = GlProgram::kFailed;
  } else {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          program.link_start)
                    .count();
    uint32_t n = ++linked_count_;
    if (n <= 8 || n % 25 == 0 || ms >= 3000.0) {
      REXGPU_INFO("GL: program #{} (vs {:016X} ps {:016X}{}) compiled+linked in {:.0f} ms on the {} thread (GLSL {} + {} bytes)",
                  n, vertex->shader().ucode_data_hash(), pixel ? pixel->shader().ucode_data_hash() : uint64_t(0),
                  program.from_disk && !program.urgent ? ", warm-up" : "", ms,
                  link_on_worker_.load() ? "compile" : "GPU", vertex->glsl()->source.size(),
                  pixel ? pixel->glsl()->source.size() : size_t(0));
    }
    if (dumped_ < 4) {
      ++dumped_;
      dump("vs_ok", vertex->glsl()->source, vertex->shader().ucode_data_hash());
      dump("vs_ucode", vertex->shader().ucode_disassembly(), vertex->shader().ucode_data_hash());
      if (pixel) {
        dump("ps_ucode", pixel->shader().ucode_disassembly(), pixel->shader().ucode_data_hash());
        dump("ps_ok", pixel->glsl()->source, pixel->shader().ucode_data_hash());
      }
    }
    program.sm_base_location = glGetUniformLocation(program.program, "xe_sm_base");
    perf::MemAdd(perf::kMemPrograms, 1);
    result = true;
  }
  glDeleteShader(program.vs_id);
  if (program.fs_id) {
    glDeleteShader(program.fs_id);
  }
  program.vs_id = program.fs_id = 0;
  return result;
}

// ---------------------------------------------------------------------------------------------
// Persistent program list (gl_program_cache)
// ---------------------------------------------------------------------------------------------

// GPU thread; mutex_ may be held.
void GlPipelineCache::RecordProgram(const GlProgram& program) {
  if (program.from_disk) return;
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (!cache_enabled_) return;
  auto put_shader = [&](const GlShader::GlTranslation* t) {
    const Shader& sh = t->shader();
    if (!cache_shaders_written_.insert(sh.ucode_data_hash()).second) return;
    Put<uint8_t>(cache_pending_, kRecShader);
    Put<uint8_t>(cache_pending_, uint8_t(sh.type()));
    Put<uint16_t>(cache_pending_, 0);
    Put<uint32_t>(cache_pending_, uint32_t(sh.ucode_dword_count()));
    Put<uint64_t>(cache_pending_, sh.ucode_data_hash());
    const size_t n = cache_pending_.size(), bytes = sh.ucode_dword_count() * 4;
    cache_pending_.resize(n + bytes);
    std::memcpy(cache_pending_.data() + n, sh.ucode_dwords(), bytes);  // host order
  };
  put_shader(program.vertex);
  if (program.pixel) put_shader(program.pixel);
  Put<uint8_t>(cache_pending_, kRecProgram);
  Put<uint8_t>(cache_pending_, 0);
  Put<uint16_t>(cache_pending_, 0);
  Put<uint32_t>(cache_pending_, 0);
  Put<uint64_t>(cache_pending_, program.vertex->shader().ucode_data_hash());
  Put<uint64_t>(cache_pending_, program.vertex->modification());
  Put<uint64_t>(cache_pending_, program.pixel ? program.pixel->shader().ucode_data_hash() : 0);
  Put<uint64_t>(cache_pending_, program.pixel ? program.pixel->modification() : 0);
}

void GlPipelineCache::FlushCacheWrites() {
  std::vector<uint8_t> data;
  std::string path;
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (!cache_enabled_ || cache_pending_.empty()) return;
    data.swap(cache_pending_);
    path = cache_path_;
  }
  if (FILE* f = std::fopen(path.c_str(), "ab")) {
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
  }
}

void GlPipelineCache::WarmUpFromDisk() {
  if (!REXCVAR_GET(gl_program_cache)) {
    REXGPU_INFO("GL program cache: off (gl_program_cache = false)");
    return;
  }
  perf::SetWorkerActivity(perf::kWWarmLoad);
  const auto t0 = std::chrono::steady_clock::now();
  const std::string path = REXCVAR_GET(gl_program_cache_path);
  std::vector<uint8_t> file;
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size > 0 && size < (64l << 20)) {
      file.resize(size_t(size));
      if (std::fread(file.data(), 1, file.size(), f) != file.size()) file.clear();
    }
    std::fclose(f);
  }
  const size_t header_size = sizeof(kCacheMagic) + 8;
  bool header_ok = file.size() >= header_size && !std::memcmp(file.data(), kCacheMagic, sizeof(kCacheMagic));
  if (header_ok) {
    uint32_t fmt = 0, tag = 0;
    std::memcpy(&fmt, file.data() + 8, 4);
    std::memcpy(&tag, file.data() + 12, 4);
    header_ok = fmt == kCacheFormat && tag == kCacheGenTag;
    if (!header_ok) {
      REXGPU_INFO("GL program cache: {} is from another translator version (format {} tag {}), starting over", path,
                  fmt, tag);
    }
  }
  if (!header_ok) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
      REXGPU_WARN("GL program cache: cannot create {} (cache off)", path);
      perf::SetWorkerActivity(perf::kWIdle);
      return;
    }
    std::fwrite(kCacheMagic, 1, sizeof(kCacheMagic), f);
    std::fwrite(&kCacheFormat, 4, 1, f);
    std::fwrite(&kCacheGenTag, 4, 1, f);
    std::fclose(f);
    file.clear();
  }
  const int warm_max = std::max(0, REXCVAR_GET(gl_program_warm_max));
  uint32_t shaders_loaded = 0, programs_queued = 0, programs_known = 0, bad = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t pos = header_ok ? header_size : file.size();
    while (pos + 16 <= file.size()) {
      const uint8_t tag = file[pos];
      if (tag == kRecShader) {
        uint8_t type = file[pos + 1];
        uint32_t count = 0;
        uint64_t hash = 0;
        std::memcpy(&count, &file[pos + 4], 4);
        std::memcpy(&hash, &file[pos + 8], 8);
        const size_t end = pos + 16 + size_t(count) * 4;
        if (count == 0 || count > 0x10000 || end > file.size()) break;  // truncated tail
        if (!shaders_.count(hash)) {
          auto shader = std::make_unique<GlShader>(type ? xenos::ShaderType::kPixel : xenos::ShaderType::kVertex, hash,
                                                   reinterpret_cast<const uint32_t*>(&file[pos + 16]), count,
                                                   std::endian::native);
          shader->AnalyzeUcode(ucode_disasm_buffer_);
          shaders_.emplace(hash, std::move(shader));
          ++shaders_loaded;
        }
        {
          std::lock_guard<std::mutex> clock(cache_mutex_);
          cache_shaders_written_.insert(hash);
        }
        pos = end;
      } else if (tag == kRecProgram) {
        if (pos + 40 > file.size()) break;
        uint64_t vh, vm, ph, pm;
        std::memcpy(&vh, &file[pos + 8], 8);
        std::memcpy(&vm, &file[pos + 16], 8);
        std::memcpy(&ph, &file[pos + 24], 8);
        std::memcpy(&pm, &file[pos + 32], 8);
        pos += 40;
        auto vit = shaders_.find(vh);
        auto pit = ph ? shaders_.find(ph) : shaders_.end();
        if (vit == shaders_.end() || (ph && pit == shaders_.end())) {
          ++bad;
          continue;
        }
        auto* vt = static_cast<GlShader::GlTranslation*>(vit->second->GetOrCreateTranslation(vm));
        auto* pt = ph ? static_cast<GlShader::GlTranslation*>(pit->second->GetOrCreateTranslation(pm)) : nullptr;
        ProgramKey key{vt, pt};
        if (programs_.count(key)) continue;
        ++programs_known;
        GlProgram& program = programs_[key];
        program.vertex = vt;
        program.pixel = pt;
        program.from_disk = true;
        if (async_ && int(programs_queued) < warm_max) {
          ++programs_queued;
          EnqueueLocked(program, false);
        }
      } else {
        ++bad;
        break;  // corrupt
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_path_ = path;
    cache_enabled_ = true;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  REXGPU_INFO("GL program cache: {} ({} bytes): {} shaders, {} programs ({} queued for background compilation{}), {} bad records, "
              "loaded in {:.0f} ms; malloc headroom {} MiB",
              path, file.size(), shaders_loaded, programs_known, programs_queued,
              link_on_worker_.load() || worker_context_ ? "" : ", translation only: no compile thread context", bad, ms,
              MallocHeadroom() >> 20);
  perf::SetWorkerActivity(perf::kWIdle);
}

}  // namespace rex::graphics::gl
