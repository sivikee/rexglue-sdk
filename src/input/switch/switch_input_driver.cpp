/**
 * @file        switch_input_driver.cpp
 * @brief       libnx HID (Npad) input driver for the Nintendo Switch homebrew port.
 *
 * Player 1 = handheld, Pro Controller or a Joy-Con pair (padInitializeDefault covers all of them).
 * Always reports one connected XInput gamepad so the guest never pauses for a missing controller.
 */

#include <rex/platform.h>

#if REX_PLATFORM_SWITCH

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/input/flags.h>
#include <rex/input/switch/switch_input_driver.h>
#include <rex/logging.h>
#include <rex/ui/virtual_key.h>

REXCVAR_DEFINE_STRING(switch_button_layout, "xbox", "Input",
                      "Switch face button layout: xbox = positional (Switch A/right = Xbox B, Switch "
                      "B/bottom = Xbox A, Switch X/top = Xbox Y, Switch Y/left = Xbox X); nintendo = "
                      "by label (Switch A = Xbox A, ...)")
    .allowed({"xbox", "nintendo"});
REXCVAR_DEFINE_STRING(switch_input_script, "", "Input",
                      "Debug: scripted button presses merged into the pad, \"swap:button:hold,...\" "
                      "(swap = GPU swap number, hold = swaps the guest has seen it pressed; \"+N:...\" = N swaps after "
                      "the guest saw the previous press released), e.g. \"1700:start:30,+300:a:10\". "
                      "Buttons (Xbox names): a b x y start back lb rb lt rt up down left right "
                      "lthumb rthumb lsup lsdown lsleft lsright");

namespace rex::input::switchpad {

std::atomic<uint32_t> g_swap_counter{0};

namespace {
constexpr DeviceId kSwitchDevice = static_cast<DeviceId>(0x53570001);
constexpr int kThumbThreshold = 0x4E00;  // same as the SDL driver
constexpr int kTriggerThreshold = 0x1F;
constexpr uint32_t kRepeatDelay = 400, kRepeatRate = 100;

uint64_t AnalogToKeyfield(const X_INPUT_GAMEPAD& g) {
  uint64_t r = 0;
  if (g.left_trigger > kTriggerThreshold) r |= 1ull << 16;
  if (g.right_trigger > kTriggerThreshold) r |= 1ull << 17;
  auto stick = [&](int x, int y, int base) {
    bool up = y > kThumbThreshold, down = y < -kThumbThreshold;
    bool right = x > kThumbThreshold, left = x < -kThumbThreshold;
    if (up && !left && !right) r |= 1ull << (base + 0);
    if (down && !left && !right) r |= 1ull << (base + 1);
    if (right && !up && !down) r |= 1ull << (base + 2);
    if (left && !up && !down) r |= 1ull << (base + 3);
    if (up && left) r |= 1ull << (base + 4);
    if (up && right) r |= 1ull << (base + 5);
    if (down && right) r |= 1ull << (base + 6);
    if (down && left) r |= 1ull << (base + 7);
  };
  stick(g.thumb_lx, g.thumb_ly, 18);
  stick(g.thumb_rx, g.thumb_ry, 26);
  return r;
}

using VK = rex::ui::VirtualKey;
constexpr std::array<VK, 34> kVkLookup = {
    VK::kXInputPadDpadUp,        VK::kXInputPadDpadDown,       VK::kXInputPadDpadLeft,
    VK::kXInputPadDpadRight,     VK::kXInputPadStart,          VK::kXInputPadBack,
    VK::kXInputPadLThumbPress,   VK::kXInputPadRThumbPress,    VK::kXInputPadLShoulder,
    VK::kXInputPadRShoulder,     VK::kNone,                    VK::kNone,
    VK::kXInputPadA,             VK::kXInputPadB,              VK::kXInputPadX,
    VK::kXInputPadY,             VK::kXInputPadLTrigger,       VK::kXInputPadRTrigger,
    VK::kXInputPadLThumbUp,      VK::kXInputPadLThumbDown,     VK::kXInputPadLThumbRight,
    VK::kXInputPadLThumbLeft,    VK::kXInputPadLThumbUpLeft,   VK::kXInputPadLThumbUpRight,
    VK::kXInputPadLThumbDownRight, VK::kXInputPadLThumbDownLeft, VK::kXInputPadRThumbUp,
    VK::kXInputPadRThumbDown,    VK::kXInputPadRThumbRight,    VK::kXInputPadRThumbLeft,
    VK::kXInputPadRThumbUpLeft,  VK::kXInputPadRThumbUpRight,  VK::kXInputPadRThumbDownRight,
    VK::kXInputPadRThumbDownLeft,
};
}  // namespace

SwitchInputDriver::SwitchInputDriver(rex::ui::Window* window, size_t window_z_order)
    : InputDriver(window, window_z_order) {}
SwitchInputDriver::~SwitchInputDriver() = default;

X_STATUS SwitchInputDriver::Setup() {
  padConfigureInput(1, HidNpadStyleSet_NpadStandard);
  padInitializeDefault(&pad_);  // player 1 + handheld
  ParseScript();
  REXLOG_INFO("input: libnx HID driver ready (button layout '{}', {} scripted presses)",
              REXCVAR_GET(switch_button_layout), script_.size());
  return X_STATUS_SUCCESS;
}

void SwitchInputDriver::ParseScript() {
  script_.clear();
  std::string s = REXCVAR_GET(switch_input_script);
  size_t pos = 0;
  while (pos < s.size()) {
    size_t end = s.find(',', pos);
    if (end == std::string::npos) end = s.size();
    std::string item = s.substr(pos, end - pos);
    pos = end + 1;
    size_t c1 = item.find(':');
    size_t c2 = c1 == std::string::npos ? c1 : item.find(':', c1 + 1);
    if (c2 == std::string::npos) continue;
    ScriptEntry e{};
    e.relative = !item.empty() && item[0] == '+';
    e.start_swap = uint32_t(std::strtoul(item.c_str() + (e.relative ? 1 : 0), nullptr, 10));
    e.seen_swap = e.released_swap = UINT32_MAX;
    std::string name = item.substr(c1 + 1, c2 - c1 - 1);
    for (auto& ch : name) ch = char(std::tolower((unsigned char)ch));
    e.hold_swaps = uint32_t(std::strtoul(item.c_str() + c2 + 1, nullptr, 10));
    if (e.hold_swaps == 0) e.hold_swaps = 1;
    struct {
      const char* n;
      uint16_t b;
    } names[] = {{"a", X_INPUT_GAMEPAD_A},
                 {"b", X_INPUT_GAMEPAD_B},
                 {"x", X_INPUT_GAMEPAD_X},
                 {"y", X_INPUT_GAMEPAD_Y},
                 {"start", X_INPUT_GAMEPAD_START},
                 {"back", X_INPUT_GAMEPAD_BACK},
                 {"lb", X_INPUT_GAMEPAD_LEFT_SHOULDER},
                 {"rb", X_INPUT_GAMEPAD_RIGHT_SHOULDER},
                 {"up", X_INPUT_GAMEPAD_DPAD_UP},
                 {"down", X_INPUT_GAMEPAD_DPAD_DOWN},
                 {"left", X_INPUT_GAMEPAD_DPAD_LEFT},
                 {"right", X_INPUT_GAMEPAD_DPAD_RIGHT},
                 {"lthumb", X_INPUT_GAMEPAD_LEFT_THUMB},
                 {"rthumb", X_INPUT_GAMEPAD_RIGHT_THUMB}};
    bool ok = false;
    for (auto& n : names) {
      if (name == n.n) {
        e.buttons = n.b;
        ok = true;
      }
    }
    if (name == "lt") { e.lt = 255; ok = true; }
    if (name == "rt") { e.rt = 255; ok = true; }
    if (name == "lsup") { e.ly = 32767; ok = true; }
    if (name == "lsdown") { e.ly = -32767; ok = true; }
    if (name == "lsleft") { e.lx = -32767; ok = true; }
    if (name == "lsright") { e.lx = 32767; ok = true; }
    if (ok) script_.push_back(e);
  }
}

void SwitchInputDriver::EnumerateDevices(std::vector<DeviceInfo>& out) {
  DeviceInfo info;
  info.id = kSwitchDevice;
  info.name = "Nintendo Switch Npad (player 1 / handheld)";
  info.guid = "switch-npad-1";
  info.synthetic = false;
  out.push_back(info);
}

X_RESULT SwitchInputDriver::GetDeviceCapabilities(DeviceId id, uint32_t,
                                                  X_INPUT_CAPABILITIES* caps) {
  if (id != kSwitchDevice) return X_ERROR_DEVICE_NOT_CONNECTED;
  if (caps) {
    std::memset(caps, 0, sizeof(*caps));
    caps->type = XINPUT_DEVTYPE_GAMEPAD;
    caps->sub_type = 0x01;  // XINPUT_DEVSUBTYPE_GAMEPAD
    caps->flags = 0;
    caps->gamepad.buttons = 0xF3FF;
    caps->gamepad.left_trigger = 0xFF;
    caps->gamepad.right_trigger = 0xFF;
    caps->gamepad.thumb_lx = int16_t(0x7FFF);
    caps->gamepad.thumb_ly = int16_t(0x7FFF);
    caps->gamepad.thumb_rx = int16_t(0x7FFF);
    caps->gamepad.thumb_ry = int16_t(0x7FFF);
    caps->vibration.left_motor_speed = 0;  // rumble not implemented
    caps->vibration.right_motor_speed = 0;
  }
  return X_ERROR_SUCCESS;
}

void SwitchInputDriver::PollLocked(X_INPUT_GAMEPAD* out) {
  padUpdate(&pad_);
  u64 h = padGetButtons(&pad_);
  uint16_t b = 0;
  bool nintendo = REXCVAR_GET(switch_button_layout) == "nintendo";
  auto map = [&](u64 mask, uint16_t xbit) {
    if (h & mask) b |= xbit;
  };
  if (nintendo) {
    map(HidNpadButton_A, X_INPUT_GAMEPAD_A);
    map(HidNpadButton_B, X_INPUT_GAMEPAD_B);
    map(HidNpadButton_X, X_INPUT_GAMEPAD_X);
    map(HidNpadButton_Y, X_INPUT_GAMEPAD_Y);
  } else {  // positional: Switch A (right) = Xbox B, B (bottom) = Xbox A, X (top) = Y, Y (left) = X
    map(HidNpadButton_A, X_INPUT_GAMEPAD_B);
    map(HidNpadButton_B, X_INPUT_GAMEPAD_A);
    map(HidNpadButton_X, X_INPUT_GAMEPAD_Y);
    map(HidNpadButton_Y, X_INPUT_GAMEPAD_X);
  }
  map(HidNpadButton_Plus, X_INPUT_GAMEPAD_START);
  map(HidNpadButton_Minus, X_INPUT_GAMEPAD_BACK);
  map(HidNpadButton_L, X_INPUT_GAMEPAD_LEFT_SHOULDER);
  map(HidNpadButton_R, X_INPUT_GAMEPAD_RIGHT_SHOULDER);
  map(HidNpadButton_StickL, X_INPUT_GAMEPAD_LEFT_THUMB);
  map(HidNpadButton_StickR, X_INPUT_GAMEPAD_RIGHT_THUMB);
  map(HidNpadButton_Up, X_INPUT_GAMEPAD_DPAD_UP);
  map(HidNpadButton_Down, X_INPUT_GAMEPAD_DPAD_DOWN);
  map(HidNpadButton_Left, X_INPUT_GAMEPAD_DPAD_LEFT);
  map(HidNpadButton_Right, X_INPUT_GAMEPAD_DPAD_RIGHT);
  HidAnalogStickState l = padGetStickPos(&pad_, 0), r = padGetStickPos(&pad_, 1);
  int lx = l.x, ly = l.y, rx = r.x, ry = r.y;  // libnx: -32767..32767, +y = up (same as XInput)
  uint8_t lt = (h & HidNpadButton_ZL) ? 255 : 0, rt = (h & HidNpadButton_ZR) ? 255 : 0;

  // A press stays down until a guest poll has seen it for hold_swaps swaps (presses during a loading screen that
  // does not poll are not lost); "+N" entries start N swaps after the previous entry's release was seen.
  uint32_t swaps = g_swap_counter.load(std::memory_order_relaxed);
  for (size_t i = 0; i < script_.size(); ++i) {
    ScriptEntry& e = script_[i];
    if (e.released_swap != UINT32_MAX) continue;
    uint32_t start = e.start_swap;
    if (e.relative) {
      const uint32_t prev = i ? script_[i - 1].released_swap : 0;
      if (prev == UINT32_MAX) continue;
      start += prev;
    }
    if (swaps < start) continue;
    if (e.seen_swap == UINT32_MAX) e.seen_swap = swaps;
    if (swaps >= e.seen_swap + e.hold_swaps) {
      e.released_swap = swaps;
      continue;
    }
    {
      b |= e.buttons;
      lt = std::max(lt, e.lt);
      rt = std::max(rt, e.rt);
      if (e.lx) lx = e.lx;
      if (e.ly) ly = e.ly;
    }
  }
  out->buttons = b;
  out->left_trigger = lt;
  out->right_trigger = rt;
  out->thumb_lx = int16_t(std::clamp(lx, -32767, 32767));
  out->thumb_ly = int16_t(std::clamp(ly, -32767, 32767));
  out->thumb_rx = int16_t(std::clamp(rx, -32767, 32767));
  out->thumb_ry = int16_t(std::clamp(ry, -32767, 32767));
}

X_RESULT SwitchInputDriver::GetDeviceState(DeviceId id, X_INPUT_STATE* out_state) {
  if (id != kSwitchDevice) return X_ERROR_DEVICE_NOT_CONNECTED;
  std::lock_guard<std::mutex> lock(mutex_);
  X_INPUT_GAMEPAD g{};
  PollLocked(&g);
  if (!have_last_ || std::memcmp(&g, &last_, sizeof(g)) != 0) {
    ++packet_number_;
    if (!have_last_ || g.buttons != last_.buttons) {  // logged on button changes only (sticks jitter)
      REXLOG_INFO("input: guest pad buttons={:04X} lt={} rt={} (swap {}, packet {})", uint16_t(g.buttons),
                  g.left_trigger, g.right_trigger, g_swap_counter.load(), packet_number_);
    }
    last_ = g;
    have_last_ = true;
  }
  if (out_state) {
    out_state->packet_number = packet_number_;
    out_state->gamepad = g;
  }
  return X_ERROR_SUCCESS;
}

X_RESULT SwitchInputDriver::SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION*) {
  return id == kSwitchDevice ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;  // accepted, ignored
}

X_RESULT SwitchInputDriver::GetDeviceKeystroke(DeviceId id, uint32_t, X_INPUT_KEYSTROKE* ks) {
  if (id != kSwitchDevice) return X_ERROR_DEVICE_NOT_CONNECTED;
  if (!ks) return X_ERROR_BAD_ARGUMENTS;
  std::lock_guard<std::mutex> lock(mutex_);
  X_INPUT_GAMEPAD g{};
  PollLocked(&g);
  const uint64_t curr = uint64_t(uint16_t(g.buttons)) | AnalogToKeyfield(g);
  auto fill = [&](int i, uint16_t flags) {
    ks->virtual_key = uint16_t(kVkLookup[i]);
    ks->unicode = 0;
    ks->flags = flags;
    ks->user_index = 0;
    ks->hid_code = 0;
  };
  uint32_t now = uint32_t(rex::chrono::Clock::QueryGuestUptimeMillis());
  if (ks_repeat_idx_ >= 0) {
    if (!ks_repeating_ && ks_repeat_time_ + kRepeatDelay < now) ks_repeating_ = true;
    if (ks_repeating_ && ks_repeat_time_ + kRepeatRate < now) {
      ks_repeat_time_ = now;
      fill(ks_repeat_idx_, X_INPUT_KEYSTROKE_KEYDOWN | X_INPUT_KEYSTROKE_REPEAT);
      return X_ERROR_SUCCESS;
    }
  }
  uint64_t changed = curr ^ ks_buttons_;
  if (!changed) return X_ERROR_EMPTY;
  for (int pass = 0; pass < 2; ++pass) {  // ups first, then downs
    for (int i = 0; i < int(kVkLookup.size()); ++i) {
      uint64_t bit = 1ull << i;
      if (!(changed & bit) || kVkLookup[i] == VK::kNone) continue;
      bool pressed = curr & bit;
      if (pass == 0 && !pressed) {
        fill(i, X_INPUT_KEYSTROKE_KEYUP);
        ks_buttons_ &= ~bit;
        ks_repeat_idx_ = -1;
        ks_repeating_ = false;
        return X_ERROR_SUCCESS;
      }
      if (pass == 1 && pressed) {
        fill(i, X_INPUT_KEYSTROKE_KEYDOWN);
        ks_buttons_ |= bit;
        ks_repeat_idx_ = i;
        ks_repeating_ = false;
        ks_repeat_time_ = now;
        return X_ERROR_SUCCESS;
      }
    }
  }
  return X_ERROR_EMPTY;
}

}  // namespace rex::input::switchpad

#endif  // REX_PLATFORM_SWITCH
