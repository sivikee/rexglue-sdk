// Nintendo Switch host fibers.
//
// XThread::Execute converts every guest thread into a "thread fiber" via
// ConvertCurrentThread(), so that must work. Real fibers (Create/SwitchTo) are only needed by
// guests using the Xbox 360 fiber exports (KeFiber-style); Oblivion imports none of them, so
// those stay fatal. Implementing them needs an aarch64 asm context swap (x19-x30, sp, d8-d15):
// newlib has no ucontext.
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/thread/fiber.h>

namespace rex::thread {

thread_local Fiber* Fiber::tls_current_ = nullptr;

Fiber* Fiber::ConvertCurrentThread() {
  if (!tls_current_) {
    tls_current_ = new Fiber();
  }
  return tls_current_;
}
Fiber* Fiber::Create(size_t, void (*)(void*), void*) {
  REX_FATAL("Fiber::Create: guest fibers are not implemented on Switch");
}
void Fiber::SwitchTo(Fiber*) {
  REX_FATAL("Fiber::SwitchTo: guest fibers are not implemented on Switch");
}
void Fiber::Destroy() {
  if (tls_current_ == this) {
    tls_current_ = nullptr;
  }
  delete this;
}

}  // namespace rex::thread
