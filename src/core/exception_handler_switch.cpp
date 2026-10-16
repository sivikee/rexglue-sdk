// Nintendo Switch host exception handling (libnx).
//
// libnx calls __libnx_exception_handler on a dedicated exception stack when a thread
// faults. Resuming is only possible by returning from the exception with the faulting
// instruction unchanged (svcReturnFromException(0)) - registers cannot be patched, so the
// POSIX-style "handler rewrites the thread context and continues" scheme (used for MMIO
// and write-watch) is NOT available. What is implemented:
//  * switch_fault_is_fatal (default true): the first guest/host fault is logged (far, pc, lr, guest
//    r1/r3.. and a guest stack walk) to sdmc:/oblivion/crash.txt and the process exits. Real hardware
//    (hbloader-hosted app) cannot resume: svcReturnFromException raises a new BadSVC (0x301)
//    exception there, so the design never relies on fault-and-resume; all guest memory is committed
//    eagerly (memory_switch.cpp).
//  * switch_fault_is_fatal=false (Eden only): legacy demand commit + resume, fault counted/logged.
//  * Any second exception (incl. BadSVC) exits immediately.
// ExceptionHandler::Install() therefore only records the handlers (never invoked).
#include <cstdio>
#include <atomic>
#include <cstdint>
#include <cstring>

#include <switch.h>

#include <rex/cvar.h>
#include <rex/exception_handler.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>
#include <rex/system/thread_state.h>

REXCVAR_DEFINE_BOOL(switch_fault_is_fatal, true, "Core",
                    "Switch: a memory fault is fatal (log + exit) instead of demand-commit + resume");

namespace rex::arch {

void ExceptionHandler::Install(Handler /*fn*/, void* /*data*/) {
  REXLOG_WARN(
      "ExceptionHandler::Install: Switch only supports demand-commit faults; handler ignored");
}

void ExceptionHandler::Uninstall(Handler /*fn*/, void* /*data*/) {}

}  // namespace rex::arch

// Optional app hook (weak): detailed report to sdmc:/oblivion/crash.txt + on screen.
extern "C" void oblivion_crash_report(ThreadExceptionDump* ctx) __attribute__((weak));
extern "C" void oblivion_bootlog(const char* fmt, ...) __attribute__((weak, format(printf, 1, 2)));

namespace {
std::atomic<int> g_exc_count{0};
std::atomic<int> g_resumed_faults{0};

// Guest context + guest stack walk (best effort; guarded by rex::memory::SwitchGuestMapped).
size_t GuestReport(char* out, size_t cap) {
  size_t n = 0;
  auto add = [&](const char* fmt, auto... a) {
    if (n + 160 < cap) n += std::snprintf(out + n, cap - n, fmt, a...);
  };
  auto* ts = rex::runtime::ThreadState::Get();
  if (!ts || !ts->context()) {
    add("guest: no ThreadState bound on this thread (host thread)\n");
    return n;
  }
  auto* c = ts->context();
  add("guest thread id=%u r1=%08x r3=%08x r4=%08x r5=%08x r6=%08x r7=%08x r8=%08x r30=%08x r31=%08x ctr=%08x\n",
      ts->thread_id(), (unsigned)c->r1.u32, (unsigned)c->r3.u32, (unsigned)c->r4.u32, (unsigned)c->r5.u32,
      (unsigned)c->r6.u32, (unsigned)c->r7.u32, (unsigned)c->r8.u32, (unsigned)c->r30.u32,
      (unsigned)c->r31.u32, (unsigned)c->ctr.u32);
  uint8_t* base = static_cast<uint8_t*>(rex::memory::SwitchArenaBase());
  auto rd32 = [&](uint32_t g, uint32_t* v) {
    if (!rex::memory::SwitchGuestMapped(g) || !rex::memory::SwitchGuestMapped(g + 3)) return false;
    uint32_t x;
    std::memcpy(&x, base + g, 4);
    *v = __builtin_bswap32(x);
    return true;
  };
  uint32_t sp = c->r1.u32;
  for (int d = 0; d < 24; ++d) {
    uint32_t lr = 0, back = 0;
    if (!rd32(sp - 8, &lr)) break;
    add("  guest frame %d: sp=%08x saved_lr=%08x\n", d, sp, lr);
    if (!rd32(sp, &back) || back <= sp) break;
    sp = back;
  }
  return n;
}

[[noreturn]] void Die(ThreadExceptionDump* ctx, const char* why) {
  static char buf[6144];
  int n = std::snprintf(buf, sizeof buf,
                        "FATAL host exception (%s): desc=0x%x pc=0x%llx far=0x%llx esr=0x%x lr=0x%llx "
                        "sp=0x%llx x0=0x%llx x1=0x%llx x2=0x%llx\n",
                        why, ctx->error_desc, (unsigned long long)ctx->pc.x, (unsigned long long)ctx->far.x,
                        ctx->esr, (unsigned long long)ctx->lr.x, (unsigned long long)ctx->sp.x,
                        (unsigned long long)ctx->cpu_gprs[0].x, (unsigned long long)ctx->cpu_gprs[1].x,
                        (unsigned long long)ctx->cpu_gprs[2].x);
  n += (int)GuestReport(buf + n, sizeof buf - n);
  if (rex::memory::SwitchArenaBase()) {
    uintptr_t off = (uintptr_t)ctx->far.x - (uintptr_t)rex::memory::SwitchArenaBase();
    if (off <= 0xFFFFFFFFull)
      n += std::snprintf(buf + n, sizeof buf - n, "far is guest address 0x%08x (mapped=%d)\n", (unsigned)off,
                         (int)rex::memory::SwitchGuestMapped((uint32_t)off));
  }
  svcOutputDebugString(buf, n);
  if (oblivion_bootlog) oblivion_bootlog("FAULT: %.900s", buf);
  if (FILE* f = std::fopen("sdmc:/rex_crash.txt", "ab")) {
    std::fwrite(buf, 1, n, f);
    std::fclose(f);
  }
  if (FILE* f = std::fopen("sdmc:/oblivion/crash_guest.txt", "ab")) {
    std::fwrite(buf, 1, n, f);
    std::fclose(f);
  }
  if (oblivion_crash_report) oblivion_crash_report(ctx);  // writes sdmc:/oblivion/crash.txt (+ on screen)
  svcExitProcess();
  __builtin_unreachable();
}
}  // namespace

extern "C" void __libnx_exception_handler(ThreadExceptionDump* ctx) {
  int nth = g_exc_count.fetch_add(1) + 1;
  // BadSVC (0x301) / any second exception: we cannot resume on hardware. Exit, never loop.
  if (nth > 1 || ctx->error_desc == 0x301) {
    if (ctx->error_desc == 0x301) {
      svcOutputDebugString("BadSVC in exception handler: exiting\n", 36);
    }
    svcExitProcess();
  }
  if (REXCVAR_GET(switch_fault_is_fatal)) {
    Die(ctx, "switch_fault_is_fatal");
  }
  // Legacy (Eden only) demand-commit + resume.
  if ((ctx->error_desc == ThreadExceptionDesc_InstructionAbort || ctx->error_desc == 0x100 ||
       ctx->error_desc == ThreadExceptionDesc_Other) &&
      rex::memory::HandleGuestMemoryFault(static_cast<uintptr_t>(ctx->far.x))) {
    g_resumed_faults.fetch_add(1);
    if (oblivion_bootlog) oblivion_bootlog("fault resumed (demand commit) far=0x%llx pc=0x%llx", (unsigned long long)ctx->far.x, (unsigned long long)ctx->pc.x);
    g_exc_count.store(0);
    svcReturnFromException(0);
  }
  Die(ctx, "unhandled");
}
