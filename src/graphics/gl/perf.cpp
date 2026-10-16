#include "perf.h"

#if REX_PLATFORM_SWITCH
#include <algorithm>
#include <climits>
#include <cstdio>
#include <deque>
#include <vector>

#include <glad/glad.h>

#include <rex/logging.h>

#include "gl_cvars.h"

namespace rex::graphics {
// command_processor.cpp (base): GPU thread idle / WAIT_REG_MEM time, guest VdSwap calls.
extern std::atomic<uint64_t> g_switch_cp_idle_ticks, g_switch_cp_waitmem_ticks, g_switch_cp_waitmem_count;
extern std::atomic<uint32_t> g_switch_vdswap_count;
}  // namespace rex::graphics

namespace rex::graphics::gl::perf {

namespace {
std::string g_lines[3];
uint64_t g_last_report = 0, g_last_swap = 0, g_sum_ft = 0, g_max_ft = 0, g_frames = 0;
uint64_t g_last_thread_cpu = 0, g_last_vdswaps = 0;
FILE* g_file = nullptr;
bool g_file_tried = false;
uint64_t g_report_ticks = 0;  // time the previous report took (log + file write)

double Ms(uint64_t ticks) { return double(armTicksToNs(ticks)) / 1e6; }
uint64_t NowNs() { return armTicksToNs(armGetSystemTick()); }

// ---- GPU timer queries ----
constexpr int kPool = 512;
bool g_timers = false;
GLuint g_q[kPool];
uint64_t g_q_cpu_ns[kPool];  // CPU time each timestamp was issued
std::vector<int> g_free;
enum RecKind { kRecSeg, kRecLag, kRecFrameEnd };
struct Rec {
  RecKind kind;
  int a, b;  // query handles (b = end, -1 when single)
  GpuSeg seg;
  uint64_t aux_cpu_ns;  // frame end: CPU time the next frame acquired its back buffer
};
std::deque<Rec> g_recs;
int64_t g_min_offset = INT64_MAX;  // min(gpu_ns - cpu_ns) over all timestamps: clock offset when the GPU is idle
// Per-report accumulators.
uint64_t g_seg_ns[kGpuSegCount], g_seg_n[kGpuSegCount];
uint64_t g_lag_sum = 0, g_lag_max = 0, g_lag_n = 0;
int64_t g_pw_sum = 0, g_pw_max = INT64_MIN, g_pw_n = 0;  // present wait beyond GPU completion
uint64_t g_lag_prev_gpu = 0, g_lag_prev_cpu = 0, g_clk_gpu = 0, g_clk_cpu = 0;
Rec* g_last_frame_end = nullptr;  // newest frame end record waiting for aux_cpu_ns
uint64_t g_stamp_fail = 0;
// ns per raw GL_TIMESTAMP unit. Mesa nouveau on the Switch returns the raw GPU timer, which ticks at 614.4 MHz
// (19.2 MHz * 32; yuzu/Eden emulate it as ns * 384 / 625), not in ns: unscaled, every GPU time was reported
// 0.6144x too small ("clk ratio" ~0.61 in v0.1.10) and the GPU/CPU offset drifted (bogus lag / present wait).
double g_ts_scale = 625.0 / 384.0;
uint64_t ScaleTs(uint64_t raw) { return uint64_t(double(raw) * g_ts_scale); }

// ---- frame fences ----
struct FrameFence {
  GLsync sync;
  uint64_t cpu_ns;
};
std::deque<FrameFence> g_fences;
uint64_t g_fl_sum = 0, g_fl_max = 0, g_fl_n = 0, g_inflight_sum = 0, g_inflight_n = 0;

int Take() {
  if (g_free.empty()) {
    ++g_stamp_fail;
    return -1;
  }
  int h = g_free.back();
  g_free.pop_back();
  return h;
}
void Give(int h) {
  if (h >= 0) g_free.push_back(h);
}
}  // namespace

const std::string* OverlayLines() { return g_lines; }

extern "C" uint32_t rex_switch_gpu_clock_mhz() __attribute__((weak));

uint64_t ThreadCpuTicks() {
  static bool failed = false;
  if (failed) return 0;
  u64 t = 0;
  Result rc = svcGetInfo(&t, InfoType_ThreadTickCount, CUR_THREAD_HANDLE, UINT64_MAX);
  if (R_FAILED(rc)) {
    failed = true;
    REXGPU_WARN("perf: svcGetInfo(ThreadTickCount) failed rc=0x{:X}; thread CPU times unavailable", rc);
    return 0;
  }
  return t;
}

void GpuTimersInit() {
  g_timers = false;
  if (!REXCVAR_GET(gl_gpu_timers)) {
    REXGPU_INFO("perf: GPU timer queries disabled (gl_gpu_timers=false)");
    return;
  }
  GLint bits = 0;
  glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &bits);
  while (glGetError() != GL_NO_ERROR) {}
  if (bits <= 0) {
    REXGPU_WARN("perf: GL_TIMESTAMP has {} counter bits, GPU timers disabled", bits);
    return;
  }
  const double sc = REXCVAR_GET(gl_gpu_timestamp_scale);
  g_ts_scale = sc > 0.0 ? sc : 625.0 / 384.0;
  glGenQueries(kPool, g_q);
  g_free.clear();
  for (int i = kPool - 1; i >= 0; --i) g_free.push_back(i);
  g_timers = true;
  REXGPU_INFO("perf: GPU timer queries enabled (GL_TIMESTAMP {} bits, pool {}, {:.6f} ns per tick)", bits, kPool, g_ts_scale);
}

int GpuStamp() {
  if (!g_timers) return -1;
  int h = Take();
  if (h < 0) return -1;
  glQueryCounter(g_q[h], GL_TIMESTAMP);
  g_q_cpu_ns[h] = NowNs();
  return h;
}

void GpuSegEnd(GpuSeg seg, int begin) {
  if (!g_timers || begin < 0) return;
  int e = GpuStamp();
  if (e < 0) {
    Give(begin);  // never issued an end: the begin query result is simply dropped
    return;
  }
  g_recs.push_back(Rec{kRecSeg, begin, e, seg, 0});
}

void GpuLagMark() {
  int h = GpuStamp();
  if (h < 0) return;
  g_recs.push_back(Rec{kRecLag, h, -1, kGpuSegFrame, 0});
}

void GpuFrameEndMark() {
  int h = GpuStamp();
  if (h < 0) return;
  g_recs.push_back(Rec{kRecFrameEnd, h, -1, kGpuSegFrame, 0});
  g_last_frame_end = &g_recs.back();  // deque::push_back keeps references valid
}

void NotePresentAcquired() {
  if (g_last_frame_end) {
    g_last_frame_end->aux_cpu_ns = NowNs();
    g_last_frame_end = nullptr;
  }
}

void GpuPoll() {
  if (!g_timers) return;
  Scope s(kPerfQueryTicks);
  while (!g_recs.empty()) {
    Rec& r = g_recs.front();
    if (r.kind == kRecFrameEnd && &r == g_last_frame_end) break;  // still waiting for the next frame's acquire
    int last = r.b >= 0 ? r.b : r.a;
    GLuint avail = 0;
    glGetQueryObjectuiv(g_q[last], GL_QUERY_RESULT_AVAILABLE, &avail);
    if (!avail) break;
    GLuint64 ta = 0, tb = 0;
    glGetQueryObjectui64v(g_q[r.a], GL_QUERY_RESULT, &ta);
    ta = ScaleTs(ta);
    int64_t off_a = int64_t(ta) - int64_t(g_q_cpu_ns[r.a]);
    g_min_offset = std::min(g_min_offset, off_a);
    if (r.kind == kRecSeg) {
      glGetQueryObjectui64v(g_q[r.b], GL_QUERY_RESULT, &tb);
      tb = ScaleTs(tb);
      g_min_offset = std::min(g_min_offset, int64_t(tb) - int64_t(g_q_cpu_ns[r.b]));
      if (tb >= ta) {
        g_seg_ns[r.seg] += tb - ta;
        ++g_seg_n[r.seg];
      }
    } else if (r.kind == kRecLag) {
      uint64_t lag = uint64_t(off_a - g_min_offset);
      g_lag_sum += lag;
      g_lag_max = std::max(g_lag_max, lag);
      ++g_lag_n;
      if (g_lag_prev_gpu && ta > g_lag_prev_gpu && g_q_cpu_ns[r.a] > g_lag_prev_cpu) {
        g_clk_gpu += ta - g_lag_prev_gpu;
        g_clk_cpu += g_q_cpu_ns[r.a] - g_lag_prev_cpu;
      }
      g_lag_prev_gpu = ta;
      g_lag_prev_cpu = g_q_cpu_ns[r.a];
    } else if (r.aux_cpu_ns) {
      // GPU finished the frame at (ta - offset) in CPU time; the next back buffer was available at aux.
      int64_t done_cpu = int64_t(ta) - g_min_offset;
      int64_t pw = int64_t(r.aux_cpu_ns) - done_cpu;
      g_pw_sum += pw;
      g_pw_max = std::max(g_pw_max, pw);
      ++g_pw_n;
    }
    Give(r.a);
    Give(r.b);
    g_recs.pop_front();
  }
  // Safety: never let the record list grow without bound (e.g. results never become available).
  while (g_recs.size() > size_t(kPool)) {
    Rec& r = g_recs.front();
    if (&r == g_last_frame_end) g_last_frame_end = nullptr;
    Give(r.a);
    Give(r.b);
    g_recs.pop_front();
  }
}

void FrameFenceIssue() {
  GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (!s) return;
  g_fences.push_back(FrameFence{s, NowNs()});
  while (g_fences.size() > 8) {
    glDeleteSync(g_fences.front().sync);
    g_fences.pop_front();
  }
}

void FrameFencePoll() {
  Scope s(kPerfQueryTicks);
  while (!g_fences.empty()) {
    GLenum r = glClientWaitSync(g_fences.front().sync, 0, 0);
    if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) break;
    uint64_t lat = NowNs() - g_fences.front().cpu_ns;
    g_fl_sum += lat;
    g_fl_max = std::max(g_fl_max, lat);
    ++g_fl_n;
    glDeleteSync(g_fences.front().sync);
    g_fences.pop_front();
  }
}

void FenceWaitLogged(void* sync, const char* who) {
  GLsync s = static_cast<GLsync>(sync);
  Scope wait_scope(kFenceWaitTicks);
  GLenum r = glClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
  int seconds = 0;
  while (r == GL_TIMEOUT_EXPIRED) {
    r = glClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
    if (r == GL_TIMEOUT_EXPIRED) {
      ++seconds;
      if (seconds == 1 || (seconds % 10) == 0) {
        REXGPU_WARN("GL: fence wait in {} has taken {} s (GPU hang?)", who, seconds);
      }
    }
  }
  if (r == GL_WAIT_FAILED) {
    REXGPU_WARN("GL: fence wait in {} failed", who);
  }
}

namespace {
void Report(uint64_t now, bool swapless);
}  // namespace

void OnSwap() {
  uint64_t now = armGetSystemTick();
  if (g_last_swap) {
    uint64_t ft = now - g_last_swap;
    g_sum_ft += ft;
    g_max_ft = std::max(g_max_ft, ft);
    ++g_frames;
  }
  g_last_swap = now;
  // Frames still in flight on the GPU when the next one starts presenting.
  FrameFencePoll();
  g_inflight_sum += g_fences.size();
  ++g_inflight_n;
  if (!g_last_report) {
    g_last_report = now;
    g_last_thread_cpu = ThreadCpuTicks();
    g_last_vdswaps = g_switch_vdswap_count.load(std::memory_order_relaxed);
    return;
  }
  if (armTicksToNs(now - g_last_report) < 1000000000ull) return;
  Report(now, false);
}

void Tick() {
  if (!g_last_report) return;  // the first swap starts the report clock
  const uint64_t now = armGetSystemTick();
  // Only when the swap-driven report is late (no swap for over a second): the frame is stalled.
  if (armTicksToNs(now - g_last_report) < 1100000000ull) return;
  Report(now, true);
}

namespace {
void Report(uint64_t now, bool swapless) {
  const uint64_t elapsed = now - g_last_report;
  uint64_t c[kCounterCount];
  for (int i = 0; i < kCounterCount; ++i) c[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
  const uint64_t cp_idle = g_switch_cp_idle_ticks.exchange(0, std::memory_order_relaxed);
  const uint64_t cp_waitmem = g_switch_cp_waitmem_ticks.exchange(0, std::memory_order_relaxed);
  const uint64_t cp_waitmem_n = g_switch_cp_waitmem_count.exchange(0, std::memory_order_relaxed);
  const uint64_t vdswaps_total = g_switch_vdswap_count.load(std::memory_order_relaxed);
  const uint64_t vdswaps = vdswaps_total - g_last_vdswaps;
  g_last_vdswaps = vdswaps_total;
  const uint64_t thread_cpu = ThreadCpuTicks();
  const uint64_t cp_cpu = thread_cpu && g_last_thread_cpu ? thread_cpu - g_last_thread_cpu : 0;
  g_last_thread_cpu = thread_cpu;
  const uint64_t report_t0 = armGetSystemTick();

  double secs = double(armTicksToNs(elapsed)) / 1e9;
  double fps = g_frames / secs;
  double avg = g_frames ? Ms(g_sum_ft) / g_frames : 0.0;
  double mx = Ms(g_max_ft);
  char l1[360], l2[320], l3[160], l4[480], l5[480], l6[560], l7[640];
  std::snprintf(l1, sizeof l1,
                "perf%s: fps=%.2f frame avg=%.1f max=%.1f ms | swap=%.0f ms swapbuf=%.0f detile=%.0f | draws=%llu (%.0f ms: const up %.0f, excl uploads %.0f) "
                "copies=%llu (%.0f ms, readback %.0f ms)",
                swapless ? " (NO SWAP for >1 s, frame stalled)" : "", fps, avg, mx, Ms(c[kSwapTicks]), Ms(c[kSwapBufTicks]), Ms(c[kSwapDetileTicks]),
                (unsigned long long)c[kDraws], Ms(c[kDrawTicks]), Ms(c[kConstUpTicks]), Ms(c[kDrawTicks]) - Ms(c[kConstUpTicks]) - Ms(c[kTexUploadTicks]) - Ms(c[kSmUploadTicks]), (unsigned long long)c[kCopies], Ms(c[kCopyTicks]),
                Ms(c[kReadbackTicks]));
  std::snprintf(l2, sizeof l2,
                "perf: xlate=%llu (%.0f ms) link=%llu (%.0f ms) | tex up=%llu %.1f MiB (%.0f ms) hash %.1f MiB (%.0f ms) | "
                "sm up %.1f MiB (%.0f ms) finish %.0f ms",
                (unsigned long long)c[kXlateCount], Ms(c[kXlateTicks]), (unsigned long long)c[kLinkCount],
                Ms(c[kLinkTicks]), (unsigned long long)c[kTexUploads], c[kTexUploadBytes] / 1048576.0,
                Ms(c[kTexUploadTicks]), c[kTexHashBytes] / 1048576.0, Ms(c[kTexHashTicks]),
                c[kSmUploadBytes] / 1048576.0, Ms(c[kSmUploadTicks]), Ms(c[kFinishTicks]));
  std::snprintf(l3, sizeof l3, "perf: sm copy %.0f ms resolve convert %.0f ms gpu-copy hits tex=%llu front=%llu", Ms(c[kSmCopyTicks]), Ms(c[kResolveConvTicks]), (unsigned long long)c[kResolveGpuHits], (unsigned long long)c[kResolveFrontHits]);
  std::snprintf(l4, sizeof l4,
                "perf: gpu resolves=%llu (%.0f ms) cpu resolves=%llu superseded=%llu deferred=%llu materialized=%llu | front snap=%llu store=%llu cpu=%llu | lazy downloads=%llu %.1f MiB (%.0f ms) | "
                "flushes: tex=%llu sm=%llu front=%llu age=%llu cap=%llu sync=%llu other=%llu",
                (unsigned long long)c[kGpuResolves], Ms(c[kGpuResolveTicks]), (unsigned long long)c[kCpuResolves],
                (unsigned long long)c[kResolveSuperseded], (unsigned long long)c[kLazyDeferred],
                (unsigned long long)c[kLazyMaterialized], (unsigned long long)c[kResolveFrontHits],
                (unsigned long long)c[kGpuFrontBlits],
                (unsigned long long)c[kCpuFrontFrames], (unsigned long long)c[kLazyDownloads],
                c[kLazyDownloadBytes] / 1048576.0, Ms(c[kLazyDownloadTicks]), (unsigned long long)c[kFlushTex],
                (unsigned long long)c[kFlushSm], (unsigned long long)c[kFlushFront], (unsigned long long)c[kFlushAge],
                (unsigned long long)c[kFlushCap], (unsigned long long)c[kFlushSync], (unsigned long long)c[kFlushOther]);
  // Where the GPU thread's second went: IssueSwap sub-steps (wall), CPU actually consumed, time waiting for the
  // guest (empty ring / WAIT_REG_MEM), guest swap rate and how many guest swaps were queued ahead of the present.
  std::snprintf(l5, sizeof l5,
                "perf: swap split: pre=%.0f clear=%.0f blit=%.0f misc=%.0f overlay=%.0f swapbuf=%.0f post=%.0f ms (swap cpu %.0f ms) | "
                "gpu thread: cpu=%.0f ms idle(ring empty)=%.0f ms waitmem=%.0f ms (%llu) | guest vdswaps=%llu ahead=%.2f | "
                "fence waits %.0f ms | perf overhead: queries %.1f ms report %.1f ms",
                Ms(c[kSwapPreTicks]), Ms(c[kSwapClearTicks]), Ms(c[kSwapBlitTicks]), Ms(c[kSwapMiscTicks]),
                Ms(c[kSwapOverlayTicks]), Ms(c[kSwapBufTicks]), Ms(c[kSwapPostTicks]), Ms(c[kSwapCpuTicks]), Ms(cp_cpu),
                Ms(cp_idle), Ms(cp_waitmem), (unsigned long long)cp_waitmem_n, (unsigned long long)vdswaps,
                g_frames ? double(c[kSwapGuestAhead]) / double(g_frames) : 0.0, Ms(c[kFenceWaitTicks]),
                Ms(c[kPerfQueryTicks]),
                Ms(g_report_ticks));
  auto seg_ms = [&](int s) { return double(g_seg_ns[s]) / 1e6; };
  if (g_timers) {
    std::snprintf(l6, sizeof l6,
                  "perf: gpu timers (ms/s): frame span=%.0f (%llu) present=%.1f blit=%.1f resolve cs=%.1f (%llu) snapshot=%.1f (%llu) | "
                  "gpu lag at present avg=%.1f max=%.1f ms | present wait beyond gpu done avg=%.1f max=%.1f ms (%lld) | "
                  "frame fence latency avg=%.1f max=%.1f ms in flight avg=%.2f | clk ratio %.3f | stamp fails %llu",
                  seg_ms(kGpuSegFrame), (unsigned long long)g_seg_n[kGpuSegFrame], seg_ms(kGpuSegSwap),
                  seg_ms(kGpuSegBlit), seg_ms(kGpuSegResolve), (unsigned long long)g_seg_n[kGpuSegResolve],
                  seg_ms(kGpuSegSnapshot), (unsigned long long)g_seg_n[kGpuSegSnapshot],
                  g_lag_n ? double(g_lag_sum) / g_lag_n / 1e6 : 0.0, double(g_lag_max) / 1e6,
                  g_pw_n ? double(g_pw_sum) / g_pw_n / 1e6 : 0.0, g_pw_n ? double(g_pw_max) / 1e6 : 0.0,
                  (long long)g_pw_n, g_fl_n ? double(g_fl_sum) / g_fl_n / 1e6 : 0.0, double(g_fl_max) / 1e6,
                  g_inflight_n ? double(g_inflight_sum) / g_inflight_n : 0.0,
                  g_clk_cpu ? double(g_clk_gpu) / double(g_clk_cpu) : 0.0, (unsigned long long)g_stamp_fail);
  } else {
    std::snprintf(l6, sizeof l6,
                  "perf: gpu timers off | frame fence latency avg=%.1f max=%.1f ms in flight avg=%.2f",
                  g_fl_n ? double(g_fl_sum) / g_fl_n / 1e6 : 0.0, double(g_fl_max) / 1e6,
                  g_inflight_n ? double(g_inflight_sum) / g_inflight_n : 0.0);
  }
  std::snprintf(l7, sizeof l7,
                "perf: draw cost hist <1ms=%llu <5=%llu <20=%llu <100=%llu <1s=%llu >=1s=%llu max %.0f ms | first-use draws %llu (%.0f ms) | "
                "skipped (program not ready) %llu | compile thread: links %llu (%.0f ms) | %s",
                (unsigned long long)c[kDrawHist0], (unsigned long long)c[kDrawHist1], (unsigned long long)c[kDrawHist2],
                (unsigned long long)c[kDrawHist3], (unsigned long long)c[kDrawHist4], (unsigned long long)c[kDrawHist5],
                Ms(c[kDrawMaxTicks]), (unsigned long long)c[kFirstUseDraws], Ms(c[kFirstUseTicks]),
                (unsigned long long)c[kSkippedDraws], (unsigned long long)c[kWLinkCount], Ms(c[kWLinkTicks]),
                ActivityString().c_str());
  REXGPU_INFO("{}", l1);
  REXGPU_INFO("{}", l2);
  REXGPU_INFO("{}", l3);
  REXGPU_INFO("{}", l4);
  REXGPU_INFO("{}", l5);
  REXGPU_INFO("{}", l6);
  REXGPU_INFO("{}", l7);
  if (!g_file_tried) {
    g_file_tried = true;
    g_file = std::fopen("sdmc:/oblivion/perf.txt", "a");
  }
  if (g_file) {
    std::fprintf(g_file, "%s\n%s\n%s\n%s\n%s\n%s\n%s\n", l1, l2, l3, l4, l5, l6, l7);
    std::fflush(g_file);
  }
  char o1[64], o2[64], o3[80];
  std::snprintf(o1, sizeof o1, "FPS %.1f  MS %.1f  MAX %.0f", fps, avg, mx);
  double dpf = g_frames ? double(c[kDraws]) / g_frames : 0.0;
  const unsigned gpu_mhz = rex_switch_gpu_clock_mhz ? rex_switch_gpu_clock_mhz() : 0;
  std::snprintf(o2, sizeof o2, "DRAWS %.0f  SHD %llu  TEX %llu  GPU %uMHZ", dpf, (unsigned long long)c[kXlateCount],
                (unsigned long long)c[kTexUploads], gpu_mhz);
  // Per frame: whole swap, its first clear (back buffer acquire), GPU thread idle, GPU lag.
  const double nf = g_frames ? double(g_frames) : 1.0;
  // GPU per frame (corrected timestamps): D = frame span minus resolves/snapshots (draws + GPU idle),
  // R = compute resolves + snapshot copies, P = present (clear + front blit + overlay).
  const double g_res = (seg_ms(kGpuSegResolve) + seg_ms(kGpuSegSnapshot)) / nf;
  const double g_draw = std::max(0.0, seg_ms(kGpuSegFrame) / nf - g_res);
  if (g_timers) {
    std::snprintf(o3, sizeof o3, "SWAP %.0f CLR %.0f IDLE %.0f GPU D%.0f R%.0f P%.0f", Ms(c[kSwapTicks]) / nf,
                  Ms(c[kSwapClearTicks]) / nf, (Ms(cp_idle) + Ms(cp_waitmem)) / nf, g_draw, g_res,
                  seg_ms(kGpuSegSwap) / nf);
  } else {
    std::snprintf(o3, sizeof o3, "SWAP %.0f CLR %.0f IDLE %.0f", Ms(c[kSwapTicks]) / nf,
                  Ms(c[kSwapClearTicks]) / nf, (Ms(cp_idle) + Ms(cp_waitmem)) / nf);
  }
  g_lines[0] = o1;
  g_lines[1] = o2;
  g_lines[2] = o3;
  g_last_report = now;
  g_sum_ft = g_max_ft = g_frames = 0;
  for (int i = 0; i < kGpuSegCount; ++i) g_seg_ns[i] = g_seg_n[i] = 0;
  g_lag_sum = g_lag_max = g_lag_n = 0;
  g_pw_sum = 0;
  g_pw_max = INT64_MIN;
  g_pw_n = 0;
  g_clk_gpu = g_clk_cpu = 0;
  g_fl_sum = g_fl_max = g_fl_n = 0;
  g_inflight_sum = g_inflight_n = 0;
  g_stamp_fail = 0;
  g_report_ticks = armGetSystemTick() - report_t0;
}
}  // namespace

// ---------------------------------------------------------------------------------------------
// Current activity (written by the GPU / compile threads, read by the heartbeat thread)
// ---------------------------------------------------------------------------------------------

namespace {
std::atomic<uint32_t> g_ph{kPhIdle}, g_ph_d0{0}, g_ph_d1{0}, g_draw_index{0};
std::atomic<uint64_t> g_ph_t0{0};
std::atomic<uint32_t> g_wph{kWIdle}, g_wq_urgent{0}, g_wq_warm{0};
std::atomic<uint64_t> g_wph_t0{0}, g_wph_a{0}, g_wph_b{0};
std::atomic<uint32_t> g_slow_logged{0};
const char* const kPhaseNames[kPhCount] = {"idle", "draw", "copy/resolve", "swap", "LINK (sync, GPU thread)",
                                           "first draw with a new program", "texture hash", "texture upload",
                                           "shared memory upload", "vertex buffers", "resolve", "flush wait",
                                           "store grow", "program lookup", "bind textures"};
const char* const kWorkerNames[kWCount] = {"idle", "translate", "compile+link", "warm-up load"};
}  // namespace

Activity::Activity(Phase phase, uint32_t d0, uint32_t d1) {
  prev_phase_ = g_ph.load(std::memory_order_relaxed);
  prev_d0_ = g_ph_d0.load(std::memory_order_relaxed);
  prev_d1_ = g_ph_d1.load(std::memory_order_relaxed);
  prev_t0_ = g_ph_t0.load(std::memory_order_relaxed);
  t0_ = armGetSystemTick();
  g_ph_d0.store(d0, std::memory_order_relaxed);
  g_ph_d1.store(d1, std::memory_order_relaxed);
  g_ph_t0.store(t0_, std::memory_order_relaxed);
  g_ph.store(phase, std::memory_order_release);
}

Activity::~Activity() {
  const uint32_t phase = g_ph.load(std::memory_order_relaxed);
  const uint64_t dt = armGetSystemTick() - t0_;
  if (armTicksToNs(dt) >= 2000000000ull && phase != kPhIdle && g_slow_logged.fetch_add(1, std::memory_order_relaxed) < 200) {
    REXGPU_WARN("GL slow: {} took {:.2f} s (detail {:08X} {:08X}, draw #{})", kPhaseNames[phase < kPhCount ? phase : 0],
                Ms(dt) / 1000.0, g_ph_d0.load(), g_ph_d1.load(), g_draw_index.load());
  }
  g_ph_d0.store(prev_d0_, std::memory_order_relaxed);
  g_ph_d1.store(prev_d1_, std::memory_order_relaxed);
  g_ph_t0.store(prev_t0_, std::memory_order_relaxed);
  g_ph.store(prev_phase_, std::memory_order_release);
}

void SetCurrentDraw(uint32_t draw_index) { g_draw_index.store(draw_index, std::memory_order_relaxed); }

void SetWorkerActivity(WorkerPhase phase, uint64_t a, uint64_t b) {
  g_wph_a.store(a, std::memory_order_relaxed);
  g_wph_b.store(b, std::memory_order_relaxed);
  g_wph_t0.store(armGetSystemTick(), std::memory_order_relaxed);
  g_wph.store(phase, std::memory_order_release);
}

void SetWorkerQueue(uint32_t urgent, uint32_t warm) {
  g_wq_urgent.store(urgent, std::memory_order_relaxed);
  g_wq_warm.store(warm, std::memory_order_relaxed);
}

std::string ActivityString() {
  const uint64_t now = armGetSystemTick();
  const uint32_t ph = g_ph.load(std::memory_order_acquire);
  const uint64_t t0 = g_ph_t0.load(std::memory_order_relaxed);
  const uint32_t wph = g_wph.load(std::memory_order_acquire);
  const uint64_t wt0 = g_wph_t0.load(std::memory_order_relaxed);
  char buf[400];
  std::snprintf(buf, sizeof buf,
                "now: gpu thread %s for %.0f ms (detail %08X %08X) draw #%u | compile thread %s for %.0f ms "
                "(%016llX %016llX) queue urgent=%u warm=%u",
                kPhaseNames[ph < kPhCount ? ph : 0], t0 && ph != kPhIdle ? Ms(now - t0) : 0.0, g_ph_d0.load(),
                g_ph_d1.load(), g_draw_index.load(), kWorkerNames[wph < kWCount ? wph : 0],
                wt0 && wph != kWIdle ? Ms(now - wt0) : 0.0, (unsigned long long)g_wph_a.load(),
                (unsigned long long)g_wph_b.load(), g_wq_urgent.load(), g_wq_warm.load());
  return buf;
}

std::string MemoryString() {
  char buf[240];
  std::snprintf(buf, sizeof buf,
                "GL memory (approx): textures %.1f MiB, render targets %.1f MiB, resolve snapshots %.1f MiB, buffers %.1f MiB, "
                "linked programs %lld",
                double(g_mem[kMemTextures].load()) / 1048576.0, double(g_mem[kMemRenderTargets].load()) / 1048576.0,
                double(g_mem[kMemSnapshots].load()) / 1048576.0, double(g_mem[kMemBuffers].load()) / 1048576.0,
                (long long)g_mem[kMemPrograms].load());
  return buf;
}

}  // namespace rex::graphics::gl::perf
#endif
