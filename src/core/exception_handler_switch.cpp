// STUB (Nintendo Switch port, phase 1): host exception (fault) handler registry.
// Must eventually: hook libnx's __libnx_exception_handler (AArch64 data abort ->
// rex::arch::Exception with HostThreadContext), dispatch registered handlers, and
// resume at the patched PC. Used by MMIO handling / guest memory protection callbacks.
// Install() currently only logs; no fault will ever reach a handler.
#include <rex/exception_handler.h>
#include <rex/logging.h>
#include <rex/platform.h>

namespace rex::arch {

void ExceptionHandler::Install(Handler /*fn*/, void* /*data*/) {
  REXLOG_WARN("ExceptionHandler::Install: not implemented on Switch (stub)");
}

void ExceptionHandler::Uninstall(Handler /*fn*/, void* /*data*/) {}

}  // namespace rex::arch
