// STUB (Nintendo Switch port, phase 1): guest SEH (hardware fault -> C++ exception).
// Must eventually: install a libnx exception handler (__libnx_exception_handler) that
// converts data aborts inside SEH-protected guest code into rex::SehException,
// mirroring seh_posix.cpp's signal handler. Until then SEH scopes never trap faults;
// seh_initialize() only logs.
#include <cstdlib>

#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/platform/seh.h>

namespace rex::platform {

static thread_local SehThreadState tls_seh_state;
static thread_local bool tls_seh_active = false;

SehThreadState& seh_thread_state() {
  return tls_seh_state;
}

int seh_filter(uint32_t /*code*/, void* /*ep*/) {
  return 0;
}

[[noreturn]] void seh_rethrow() {
  REX_FATAL("seh_rethrow: SEH is not implemented on Switch");
}

void seh_initialize() {
  if (g_seh_initialized.exchange(true)) {
    return;
  }
  REXLOG_WARN("SEH fault trapping is not implemented on Switch (stub)");
}

bool& seh_active() {
  return tls_seh_active;
}

}  // namespace rex::platform
