// Lightweight per-second performance report for the Switch GL backend (no-op elsewhere).
#pragma once
#include <cstdint>
#include <string>

#include <rex/platform.h>

#if REX_PLATFORM_SWITCH
#include <atomic>
#include <switch.h>
#endif

namespace rex::graphics::gl::perf {

enum Counter : int {
  kDraws, kDrawTicks,
  kCopies, kCopyTicks, kReadbackTicks,          // resolve (EDRAM copy) incl. glReadPixels stall
  kXlateCount, kXlateTicks,                     // ucode -> SPIR-V -> GLSL (worker thread CPU time)
  kLinkCount, kLinkTicks,                       // GL compile/link issue + finish (GL thread)
  kTexUploads, kTexUploadBytes, kTexUploadTicks,
  kTexHashBytes, kTexHashTicks,
  kSmUploadBytes, kSmUploadTicks,
  kFinishTicks,
  kSwapTicks, kSwapDetileTicks, kSwapBufTicks,
  kCounterCount
};

#if REX_PLATFORM_SWITCH
inline std::atomic<uint64_t> g_counters[kCounterCount];
inline void Add(Counter c, uint64_t n = 1) { g_counters[c].fetch_add(n, std::memory_order_relaxed); }
struct Scope {
  Counter c;
  uint64_t t;
  explicit Scope(Counter c_) : c(c_), t(armGetSystemTick()) {}
  ~Scope() { Add(c, armGetSystemTick() - t); }
};
// Call once per presented frame (start of the swap). Emits a report to the log and
// sdmc:/oblivion/perf.txt about once per second.
void OnSwap();
// Text of the last report for the on-screen overlay (array of 3 strings).
const std::string* OverlayLines();
#else
inline void Add(Counter, uint64_t = 1) {}
struct Scope { explicit Scope(Counter) {} };
inline void OnSwap() {}
inline const std::string* OverlayLines() { return nullptr; }
#endif

}  // namespace rex::graphics::gl::perf
