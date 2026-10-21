#pragma once
/**
 * @file        rex/input/switch/switch_input_driver.h
 * @brief       libnx HID (Npad) input driver for the Nintendo Switch homebrew port.
 */

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <rex/input/input_driver.h>
#include <rex/platform.h>

#if REX_PLATFORM_SWITCH
#include <switch.h>

namespace rex::input::switchpad {

/// Incremented by the GPU backend on every guest swap; the `switch_input_script` cvar is keyed on it.
extern std::atomic<uint32_t> g_swap_counter;

class SwitchInputDriver final : public InputDriver {
 public:
  explicit SwitchInputDriver(rex::ui::Window* window, size_t window_z_order);
  ~SwitchInputDriver() override;

  X_STATUS Setup() override;

  void EnumerateDevices(std::vector<DeviceInfo>& out) override;
  X_RESULT GetDeviceState(DeviceId id, X_INPUT_STATE* out_state) override;
  X_RESULT GetDeviceCapabilities(DeviceId id, uint32_t flags,
                                 X_INPUT_CAPABILITIES* out_caps) override;
  X_RESULT SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION* vibration) override;
  X_RESULT GetDeviceKeystroke(DeviceId id, uint32_t flags,
                              X_INPUT_KEYSTROKE* out_keystroke) override;

 private:
  struct ScriptEntry {
    uint32_t start_swap;  // absolute swap, or swaps after the previous entry's observed release ("+N")
    bool relative;
    uint32_t hold_swaps;  // counted from the first guest poll that saw the press
    uint32_t seen_swap;      // swap of the first poll that saw it pressed (UINT32_MAX = not yet)
    uint32_t released_swap;  // swap of the first poll that saw it released (UINT32_MAX = not yet)
    uint16_t buttons;  // X_INPUT_GAMEPAD_* bits
    uint8_t lt, rt;
    int16_t lx, ly;
  };
  // Polls libnx and merges the debug script. Caller holds mutex_.
  void PollLocked(X_INPUT_GAMEPAD* out);
  void ParseScript();

  std::mutex mutex_;
  PadState pad_;
  uint32_t packet_number_ = 0;
  X_INPUT_GAMEPAD last_{};
  bool have_last_ = false;
  std::vector<ScriptEntry> script_;
  // Keystroke generation (same scheme as the SDL driver).
  uint64_t ks_buttons_ = 0;
  int ks_repeat_idx_ = -1;
  uint32_t ks_repeat_time_ = 0;
  bool ks_repeating_ = false;
};

}  // namespace rex::input::switchpad
#endif
