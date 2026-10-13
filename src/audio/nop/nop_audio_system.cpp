/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/audio/flags.h>
#include <rex/platform.h>
#include <rex/audio/nop/nop_audio_system.h>

#if REX_PLATFORM_SWITCH
#include <chrono>

#include <rex/audio/audio_driver.h>
#include <rex/thread.h>
#endif

namespace rex::audio::nop {

#if REX_PLATFORM_SWITCH
// Switch bring-up: a driver that discards the samples but paces the guest audio callback at the
// real frame rate (256 samples @ 48 kHz = 5.333 ms), like a hardware endpoint would. Without a
// driver the game treats RegisterClient failure as fatal during D3D/renderer init.
class NullAudioDriver : public AudioDriver {
 public:
  NullAudioDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore)
      : AudioDriver(memory), semaphore_(semaphore), next_(std::chrono::steady_clock::now()) {}
  void SubmitFrame(uint32_t /*samples_ptr*/) override {
    using namespace std::chrono;
    next_ += microseconds(5333);
    auto now = steady_clock::now();
    if (next_ > now) {
      rex::thread::Sleep(duration_cast<microseconds>(next_ - now));
    } else if (now - next_ > milliseconds(100)) {
      next_ = now;  // fell behind: resync instead of bursting
    }
    semaphore_->Release(1, nullptr);
  }

 private:
  rex::thread::Semaphore* semaphore_;
  std::chrono::steady_clock::time_point next_;
};
#endif

std::unique_ptr<AudioSystem> NopAudioSystem::Create(
    runtime::FunctionDispatcher* function_dispatcher) {
  return std::make_unique<NopAudioSystem>(function_dispatcher);
}

NopAudioSystem::NopAudioSystem(runtime::FunctionDispatcher* function_dispatcher)
    : AudioSystem(function_dispatcher) {}

NopAudioSystem::~NopAudioSystem() = default;

X_STATUS NopAudioSystem::CreateDriver(size_t /*index*/, rex::thread::Semaphore* semaphore,
                                      AudioDriver** out_driver) {
#if REX_PLATFORM_SWITCH
  *out_driver = new NullAudioDriver(memory(), semaphore);
  return X_STATUS_SUCCESS;
#else
  (void)semaphore;
  (void)out_driver;
  return X_STATUS_NOT_IMPLEMENTED;
#endif
}

void NopAudioSystem::DestroyDriver(AudioDriver* driver) {
#if REX_PLATFORM_SWITCH
  delete driver;
#else
  (void)driver;
  assert_always();
#endif
}

}  // namespace rex::audio::nop
