// STUB (Nintendo Switch port, phase 1): host fibers.
// Must eventually: implement stack switching on AArch64 (newlib has no ucontext;
// either a hand-written asm swap of x19-x30/sp/d8-d15 or libnx Thread based coroutines).
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/thread/fiber.h>

namespace rex::thread {

thread_local Fiber* Fiber::tls_current_ = nullptr;

Fiber* Fiber::ConvertCurrentThread() {
  REX_FATAL("Fiber::ConvertCurrentThread: not implemented on Switch");
}
Fiber* Fiber::Create(size_t, void (*)(void*), void*) {
  REX_FATAL("Fiber::Create: not implemented on Switch");
}
void Fiber::SwitchTo(Fiber*) {
  REX_FATAL("Fiber::SwitchTo: not implemented on Switch");
}
void Fiber::Destroy() {}

}  // namespace rex::thread
