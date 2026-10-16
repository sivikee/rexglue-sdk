// Nintendo Switch host exception handling (libnx).
//
// libnx calls __libnx_exception_handler on a dedicated exception stack when a thread
// faults. Resuming is only possible by returning from the exception with the faulting
// instruction unchanged (svcReturnFromException(0)) - registers cannot be patched, so the
// POSIX-style "handler rewrites the thread context and continues" scheme (used for MMIO
// and write-watch) is NOT available. What is implemented:
//  * demand commit of guest arena segments (see memory_switch.cpp) -> retry the access;
//  * anything else: log the fault (debug output + sdmc:/rex_crash.txt) and abort.
// ExceptionHandler::Install() therefore only records the handlers (never invoked).
#include <cstdio>
#include <cstring>

#include <switch.h>

#include <rex/exception_handler.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>

namespace rex::arch {

void ExceptionHandler::Install(Handler /*fn*/, void* /*data*/) {
  REXLOG_WARN(
      "ExceptionHandler::Install: Switch only supports demand-commit faults; handler ignored");
}

void ExceptionHandler::Uninstall(Handler /*fn*/, void* /*data*/) {}

}  // namespace rex::arch

// Optional app hook (weak): detailed report to sdmc:/oblivion/crash.txt + on screen.
extern "C" void oblivion_crash_report(ThreadExceptionDump* ctx) __attribute__((weak));

extern "C" void __libnx_exception_handler(ThreadExceptionDump* ctx) {
  if ((ctx->error_desc == ThreadExceptionDesc_InstructionAbort || ctx->error_desc == 0x100 ||
       ctx->error_desc == ThreadExceptionDesc_Other) &&
      rex::memory::HandleGuestMemoryFault(static_cast<uintptr_t>(ctx->far.x))) {
    svcReturnFromException(0);
  }
  if (oblivion_crash_report) oblivion_crash_report(ctx);
  char buf[512];
  int n = std::snprintf(buf, sizeof buf,
                        "FATAL host exception: desc=0x%x pc=0x%llx far=0x%llx esr=0x%x lr=0x%llx "
                        "sp=0x%llx x0=0x%llx x1=0x%llx x2=0x%llx\n",
                        ctx->error_desc, (unsigned long long)ctx->pc.x,
                        (unsigned long long)ctx->far.x, ctx->esr, (unsigned long long)ctx->lr.x,
                        (unsigned long long)ctx->sp.x, (unsigned long long)ctx->cpu_gprs[0].x,
                        (unsigned long long)ctx->cpu_gprs[1].x, (unsigned long long)ctx->cpu_gprs[2].x);
  svcOutputDebugString(buf, n);
  if (FILE* f = std::fopen("sdmc:/rex_crash.txt", "ab")) {
    std::fwrite(buf, 1, n, f);
    std::fclose(f);
  }
  svcReturnFromException(0xf801);
}
