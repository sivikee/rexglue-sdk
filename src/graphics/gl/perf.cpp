#include "perf.h"

#if REX_PLATFORM_SWITCH
#include <algorithm>
#include <cstdio>

#include <rex/logging.h>

namespace rex::graphics::gl::perf {

namespace {
std::string g_lines[3];
uint64_t g_last_report = 0, g_last_swap = 0, g_sum_ft = 0, g_max_ft = 0, g_frames = 0;
FILE* g_file = nullptr;
bool g_file_tried = false;

double Ms(uint64_t ticks) { return double(armTicksToNs(ticks)) / 1e6; }
}  // namespace

const std::string* OverlayLines() { return g_lines; }

void OnSwap() {
  uint64_t now = armGetSystemTick();
  if (g_last_swap) {
    uint64_t ft = now - g_last_swap;
    g_sum_ft += ft;
    g_max_ft = std::max(g_max_ft, ft);
    ++g_frames;
  }
  g_last_swap = now;
  if (!g_last_report) {
    g_last_report = now;
    return;
  }
  uint64_t elapsed = now - g_last_report;
  if (armTicksToNs(elapsed) < 1000000000ull) return;

  uint64_t c[kCounterCount];
  for (int i = 0; i < kCounterCount; ++i) c[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
  double secs = double(armTicksToNs(elapsed)) / 1e9;
  double fps = g_frames / secs;
  double avg = g_frames ? Ms(g_sum_ft) / g_frames : 0.0;
  double mx = Ms(g_max_ft);
  char l1[320], l2[320], l3[160], l4[400];
  std::snprintf(l1, sizeof l1,
                "perf: fps=%.2f frame avg=%.1f max=%.1f ms | swap=%.0f ms swapbuf=%.0f detile=%.0f | draws=%llu (%.0f ms: const up %.0f, excl uploads %.0f) "
                "copies=%llu (%.0f ms, readback %.0f ms)",
                fps, avg, mx, Ms(c[kSwapTicks]), Ms(c[kSwapBufTicks]), Ms(c[kSwapDetileTicks]),
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
                "perf: gpu resolves=%llu (%.0f ms) cpu resolves=%llu superseded=%llu | front gpu=%llu cpu=%llu | lazy downloads=%llu %.1f MiB (%.0f ms) | "
                "flushes: tex=%llu sm=%llu front=%llu age=%llu cap=%llu sync=%llu other=%llu",
                (unsigned long long)c[kGpuResolves], Ms(c[kGpuResolveTicks]), (unsigned long long)c[kCpuResolves],
                (unsigned long long)c[kResolveSuperseded], (unsigned long long)c[kGpuFrontBlits],
                (unsigned long long)c[kCpuFrontFrames], (unsigned long long)c[kLazyDownloads],
                c[kLazyDownloadBytes] / 1048576.0, Ms(c[kLazyDownloadTicks]), (unsigned long long)c[kFlushTex],
                (unsigned long long)c[kFlushSm], (unsigned long long)c[kFlushFront], (unsigned long long)c[kFlushAge],
                (unsigned long long)c[kFlushCap], (unsigned long long)c[kFlushSync], (unsigned long long)c[kFlushOther]);
  REXGPU_INFO("{}", l1);
  REXGPU_INFO("{}", l2);
  REXGPU_INFO("{}", l3);
  REXGPU_INFO("{}", l4);
  if (!g_file_tried) {
    g_file_tried = true;
    g_file = std::fopen("sdmc:/oblivion/perf.txt", "a");
  }
  if (g_file) {
    std::fprintf(g_file, "%s\n%s\n%s\n%s\n", l1, l2, l3, l4);
    std::fflush(g_file);
  }
  char o1[64], o2[64];
  std::snprintf(o1, sizeof o1, "FPS %.1f  MS %.1f  MAX %.0f", fps, avg, mx);
  double dpf = g_frames ? double(c[kDraws]) / g_frames : 0.0;
  std::snprintf(o2, sizeof o2, "DRAWS %.0f  SHD %llu  TEX %llu", dpf, (unsigned long long)c[kXlateCount],
                (unsigned long long)c[kTexUploads]);
  g_lines[0] = o1;
  g_lines[1] = o2;
  g_lines[2].clear();
  g_last_report = now;
  g_sum_ft = g_max_ft = g_frames = 0;
}

}  // namespace rex::graphics::gl::perf
#endif
