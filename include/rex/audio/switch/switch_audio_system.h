/**
 * Nintendo Switch audio backend (libnx audout). REX_PLATFORM_SWITCH only.
 * Uses the real AudioSystem + XMA decoder; this class only provides the output driver.
 */
#pragma once

#include <rex/audio/audio_system.h>

namespace rex::audio::switch_hw {

class SwitchAudioSystem : public AudioSystem {
 public:
  explicit SwitchAudioSystem(runtime::FunctionDispatcher* function_dispatcher);
  ~SwitchAudioSystem() override;

  static bool IsAvailable() { return true; }
  static std::unique_ptr<AudioSystem> Create(runtime::FunctionDispatcher* function_dispatcher);

  X_STATUS CreateDriver(size_t index, rex::thread::Semaphore* semaphore,
                        AudioDriver** out_driver) override;
  void DestroyDriver(AudioDriver* driver) override;
};

}  // namespace rex::audio::switch_hw
