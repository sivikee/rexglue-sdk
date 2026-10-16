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
  kConstUpTicks,                                // per-draw uniform upload (nested inside draw ticks)
  kResolveConvTicks, kResolveGpuHits, kResolveFrontHits,  // deferred resolve CPU conversion, GPU-copy consumers
  kSmCopyTicks,
  kSwapTicks, kSwapDetileTicks, kSwapBufTicks,
  // flush reasons (number of resolves forced to be written to guest memory), GPU resolve statistics
  kFlushTex, kFlushSm, kFlushFront, kFlushAge, kFlushCap, kFlushSync, kFlushOther,
  kGpuResolves, kCpuResolves, kGpuFrontBlits, kCpuFrontFrames, kLazyDownloads, kLazyDownloadBytes, kLazyDownloadTicks,
  kResolveSuperseded, kGpuResolveTicks,
  kLazyDeferred, kLazyMaterialized,             // lazy GPU resolves: compute write deferred / run later on demand
  // IssueSwap split (CPU wall time of each sub-step on the GPU thread)
  kSwapPreTicks, kSwapClearTicks, kSwapBlitTicks, kSwapMiscTicks, kSwapOverlayTicks, kSwapPostTicks,
  kSwapCpuTicks,      // CPU time the GPU thread actually consumed inside IssueSwap (svc ThreadTickCount)
  kSwapGuestAhead,    // sum over swaps of (guest VdSwap count - presented swap count)
  kPerfQueryTicks,    // GPU timer query / fence polling overhead
  kFenceWaitTicks,    // blocking CPU waits on GL fences (stream rings, resolve read-backs)
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

// ---- GPU-side timing (GL_TIMESTAMP queries) and frame completion fences; GL thread only ----
enum GpuSeg : int {
  kGpuSegFrame,     // GPU timestamp span from the end of the previous present to the start of this one (incl. idle)
  kGpuSegSwap,      // present work: clear + front buffer blit + overlay
  kGpuSegBlit,      // front buffer blit draw alone
  kGpuSegResolve,   // compute resolves (dispatch)
  kGpuSegSnapshot,  // resolve snapshot copies (glCopyImageSubData)
  kGpuSegCount
};
// Initialises the query pool (call once with a current context). Logs timer support.
void GpuTimersInit();
// Issues a timestamp; returns a handle (or -1 when disabled / pool exhausted).
int GpuStamp();
// Issues the end timestamp of a segment that began with GpuStamp(); the duration is added when available.
void GpuSegEnd(GpuSeg seg, int begin);
// Timestamp at the start of a present: measures how far the GPU runs behind the CPU (lag).
void GpuLagMark();
// Timestamp at the end of a frame's GPU work (just before eglSwapBuffers). The next NotePresentAcquired() (CPU
// time the next frame got its back buffer, i.e. after the first clear of the default framebuffer) gives the
// present wait beyond GPU completion (vsync / compositor).
void GpuFrameEndMark();
void NotePresentAcquired();
// Frame completion fence (after eglSwapBuffers) and non-blocking polling of older ones.
void FrameFenceIssue();
void FrameFencePoll();
// Collects available query results (call once per swap).
void GpuPoll();
// Waits on a fence like the backend's loops (GL_SYNC_FLUSH_COMMANDS_BIT, 1 s slices) and warns about long waits.
void FenceWaitLogged(void* sync, const char* who);
// CPU time consumed by the calling thread (svcGetInfo ThreadTickCount), in system ticks; 0 if unavailable.
uint64_t ThreadCpuTicks();
#else
inline void Add(Counter, uint64_t = 1) {}
struct Scope { explicit Scope(Counter) {} };
inline void OnSwap() {}
inline const std::string* OverlayLines() { return nullptr; }
enum GpuSeg : int { kGpuSegFrame, kGpuSegSwap, kGpuSegBlit, kGpuSegResolve, kGpuSegSnapshot, kGpuSegCount };
inline void GpuTimersInit() {}
inline int GpuStamp() { return -1; }
inline void GpuSegEnd(GpuSeg, int) {}
inline void GpuLagMark() {}
inline void GpuFrameEndMark() {}
inline void NotePresentAcquired() {}
inline void FrameFenceIssue() {}
inline void FrameFencePoll() {}
inline void GpuPoll() {}
inline void FenceWaitLogged(void*, const char*) {}
inline uint64_t ThreadCpuTicks() { return 0; }
#endif

}  // namespace rex::graphics::gl::perf
