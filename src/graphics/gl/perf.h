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
  kFrontDropped,                                // shown lazy front buffers aged out without a write-back
  // IssueSwap split (CPU wall time of each sub-step on the GPU thread)
  kSwapPreTicks, kSwapClearTicks, kSwapBlitTicks, kSwapMiscTicks, kSwapOverlayTicks, kSwapPostTicks,
  kSwapCpuTicks,      // CPU time the GPU thread actually consumed inside IssueSwap (svc ThreadTickCount)
  kSwapGuestAhead,    // sum over swaps of (guest VdSwap count - presented swap count)
  kPerfQueryTicks,    // GPU timer query / fence polling overhead
  kFenceWaitTicks,    // blocking CPU waits on GL fences (stream rings, resolve read-backs)
  // Per-draw wall time histogram (IssueDraw incl. everything it triggers): <1, <5, <20, <100, <1000, >=1000 ms.
  kDrawHist0, kDrawHist1, kDrawHist2, kDrawHist3, kDrawHist4, kDrawHist5,
  kDrawMaxTicks,      // slowest single draw of the report window (max, not sum)
  kFirstUseDraws, kFirstUseTicks,  // draws that used a program for the first time (Mesa may compile a variant)
  kWLinkCount, kWLinkTicks,        // compile + link done on the compile thread (shared context), not blocking draws
  kSkippedDraws,                   // draws skipped because their program was not ready yet
  kHandoverCount,                  // compile-thread programs that became usable on the GPU thread
  kDeadlineLinkCount, kDeadlineLinkTicks,  // programs linked on the GPU thread after waiting > gl_program_max_wait_ms
  kCounterCount
};

#if REX_PLATFORM_SWITCH
inline std::atomic<uint64_t> g_counters[kCounterCount];
inline void Add(Counter c, uint64_t n = 1) { g_counters[c].fetch_add(n, std::memory_order_relaxed); }
inline uint64_t Get(Counter c) { return g_counters[c].load(std::memory_order_relaxed); }
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
// Time-based report: call from the GL thread outside swaps (draws, copies). Emits the per-second report even
// when no swap happens (a stalled frame then still shows where the time goes).
void Tick();
// Max-style counter update (kDrawMaxTicks).
inline void Max(Counter c, uint64_t v) {
  uint64_t cur = g_counters[c].load(std::memory_order_relaxed);
  while (v > cur && !g_counters[c].compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
}

// ---- "what is the GPU thread doing right now" (read by the app heartbeat from another thread) ----
enum Phase : uint32_t {
  kPhIdle, kPhDraw, kPhCopy, kPhSwap, kPhLinkSync, kPhFirstUse, kPhTexHash, kPhTexUpload, kPhSmUpload,
  kPhVertexPrep, kPhResolve, kPhFlushWait, kPhStoreGrow, kPhProgramLookup, kPhBindTextures, kPhCount
};
// Compile thread phases.
enum WorkerPhase : uint32_t { kWIdle, kWTranslate, kWCompileLink, kWWarmLoad, kWCount };
// RAII: sets the GPU thread's current phase (with two detail words) and restores the outer one on exit.
// Logs a warning when one phase instance takes longer than 2 s.
class Activity {
 public:
  Activity(Phase phase, uint32_t d0 = 0, uint32_t d1 = 0);
  ~Activity();
 private:
  uint32_t prev_phase_, prev_d0_, prev_d1_;
  uint64_t prev_t0_, t0_;
};
void SetCurrentDraw(uint32_t draw_index);
void SetWorkerActivity(WorkerPhase phase, uint64_t a = 0, uint64_t b = 0);
void SetWorkerQueue(uint32_t urgent, uint32_t warm);
// One-line snapshot: GPU thread phase + duration + details, compile thread phase, queue sizes, draw cost histogram
// since boot. Thread-safe.
std::string ActivityString();
// GL memory accounting (bytes, approximate): textures, render targets, resolve snapshots, buffers; programs.
enum MemKind : int { kMemTextures, kMemRenderTargets, kMemSnapshots, kMemBuffers, kMemPrograms, kMemCount };
inline std::atomic<int64_t> g_mem[kMemCount];
inline void MemAdd(MemKind k, int64_t bytes) { g_mem[k].fetch_add(bytes, std::memory_order_relaxed); }
std::string MemoryString();
#else
inline void Add(Counter, uint64_t = 1) {}
inline uint64_t Get(Counter) { return 0; }
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
inline void Tick() {}
inline void Max(Counter, uint64_t) {}
enum Phase : uint32_t {
  kPhIdle, kPhDraw, kPhCopy, kPhSwap, kPhLinkSync, kPhFirstUse, kPhTexHash, kPhTexUpload, kPhSmUpload,
  kPhVertexPrep, kPhResolve, kPhFlushWait, kPhStoreGrow, kPhProgramLookup, kPhBindTextures, kPhCount
};
enum WorkerPhase : uint32_t { kWIdle, kWTranslate, kWCompileLink, kWWarmLoad, kWCount };
class Activity { public: explicit Activity(Phase, uint32_t = 0, uint32_t = 0) {} };
inline void SetCurrentDraw(uint32_t) {}
inline void SetWorkerActivity(WorkerPhase, uint64_t = 0, uint64_t = 0) {}
inline void SetWorkerQueue(uint32_t, uint32_t) {}
inline std::string ActivityString() { return {}; }
enum MemKind : int { kMemTextures, kMemRenderTargets, kMemSnapshots, kMemBuffers, kMemPrograms, kMemCount };
inline void MemAdd(MemKind, int64_t) {}
inline std::string MemoryString() { return {}; }
#endif

}  // namespace rex::graphics::gl::perf
