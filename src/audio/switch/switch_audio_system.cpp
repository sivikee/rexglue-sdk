/**
 * Nintendo Switch audio backend (libnx audout). REX_PLATFORM_SWITCH only.
 *
 * The guest mixes 6-channel planar big-endian float frames of 256 samples @ 48 kHz. SubmitFrame
 * copies a frame into a queue; an output thread folds queued frames to stereo s16, hands them to
 * audout (one 256-sample buffer each) and releases the guest semaphore once per frame that
 * reached the device (silence when the queue is empty), like the SDL driver's callback.
 *
 * Cvars: switch_audio_dump = "<path>.wav" writes the stereo PCM that is sent to audout.
 */
#include <rex/platform.h>

#if REX_PLATFORM_SWITCH

#include <switch.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <stack>
#include <string>
#include <thread>

#include <rex/audio/audio_driver.h>
#include <rex/audio/conversion.h>
#include <rex/audio/downmix.h>
#include <rex/audio/switch/switch_audio_system.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/thread.h>

REXCVAR_DEFINE_STRING(switch_audio_dump, "", "Audio",
                      "Switch: write the stereo PCM sent to audout to this WAV file");

extern "C" void oblivion_bootlog(const char* fmt, ...) __attribute__((weak, format(printf, 1, 2)));
namespace rex::audio::switch_hw {

namespace {
constexpr uint32_t kChannelSamples = 256;
constexpr uint32_t kFrameSamples = 6 * kChannelSamples;
constexpr uint32_t kRate = 48000;
constexpr uint32_t kBufferCount = 4;
constexpr uint32_t kBufferBytes = 0x1000;  // audout wants 0x1000 aligned sizes
constexpr uint32_t kDataBytes = kChannelSamples * 2 * sizeof(int16_t);

class WavWriter {
 public:
  void Open(const std::string& path) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) {
      REXAPU_ERROR("switch audio: cannot open dump '{}'", path);
      return;
    }
    WriteHeader();
  }
  void Write(const int16_t* s, size_t frames) {
    if (!f_) return;
    std::fwrite(s, 4, frames, f_);
    data_ += frames * 4;
    if (++writes_ % 188 == 0) {  // ~1 s: keep the header valid in case the process is killed
      WriteHeader();
    }
  }
  void Close() {
    if (f_) {
      WriteHeader();
      std::fclose(f_);
      f_ = nullptr;
    }
  }

 private:
  void WriteHeader() {
    uint8_t h[44];
    auto u32 = [&](int o, uint32_t v) { std::memcpy(h + o, &v, 4); };
    auto u16 = [&](int o, uint16_t v) { std::memcpy(h + o, &v, 2); };
    std::memcpy(h, "RIFF", 4);
    u32(4, static_cast<uint32_t>(36 + data_));
    std::memcpy(h + 8, "WAVEfmt ", 8);
    u32(16, 16);
    u16(20, 1);
    u16(22, 2);
    u32(24, kRate);
    u32(28, kRate * 4);
    u16(32, 4);
    u16(34, 16);
    std::memcpy(h + 36, "data", 4);
    u32(40, static_cast<uint32_t>(data_));
    long pos = std::ftell(f_);
    std::fseek(f_, 0, SEEK_SET);
    std::fwrite(h, 1, 44, f_);
    std::fseek(f_, pos < 44 ? 44 : pos, SEEK_SET);
    std::fflush(f_);
  }
  FILE* f_ = nullptr;
  uint64_t data_ = 0;
  uint32_t writes_ = 0;
};

class AudoutDriver : public AudioDriver {
 public:
  AudoutDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore)
      : AudioDriver(memory), semaphore_(semaphore) {}

  bool Initialize() {
    if (oblivion_bootlog) oblivion_bootlog("audio: audoutInitialize...");
    Result rc = audoutInitialize();
    if (oblivion_bootlog) oblivion_bootlog("audio: audoutInitialize rc=0x%x", (unsigned)rc);
    if (R_FAILED(rc)) {
      REXAPU_ERROR("audoutInitialize failed: {:#x}", rc);
      return false;
    }
    rc = audoutStartAudioOut();
    if (oblivion_bootlog) oblivion_bootlog("audio: audoutStartAudioOut rc=0x%x", (unsigned)rc);
    if (R_FAILED(rc)) {
      REXAPU_ERROR("audoutStartAudioOut failed: {:#x}", rc);
      audoutExit();
      return false;
    }
    REXAPU_INFO("switch audio: audout {} Hz, {} ch (guest 48000 Hz 6ch folded to stereo s16)",
                audoutGetSampleRate(), audoutGetChannelCount());
    for (uint32_t i = 0; i < kBufferCount; ++i) {
      mem_[i] = aligned_alloc(0x1000, kBufferBytes);
      std::memset(mem_[i], 0, kBufferBytes);
      bufs_[i] = {};
      bufs_[i].buffer = mem_[i];
      bufs_[i].buffer_size = kBufferBytes;
      bufs_[i].data_size = kDataBytes;
    }
    std::string dump = REXCVAR_GET(switch_audio_dump);
    if (!dump.empty()) {
      wav_.Open(dump);
      REXAPU_INFO("switch audio: dumping to {}", dump);
    }
    running_ = true;
    thread_ = std::thread([this] { Run(); });
    return true;
  }

  void Shutdown() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
    audoutStopAudioOut();
    audoutExit();
    wav_.Close();
    for (auto& m : mem_) {
      std::free(m);
      m = nullptr;
    }
    std::lock_guard<std::mutex> g(mutex_);
    while (!queued_.empty()) {
      delete[] queued_.front();
      queued_.pop();
    }
    while (!unused_.empty()) {
      delete[] unused_.top();
      unused_.pop();
    }
  }

  void SubmitFrame(uint32_t frame_ptr) override {
    const auto in = memory_->TranslateVirtual<float*>(frame_ptr);
    float* out;
    {
      std::lock_guard<std::mutex> g(mutex_);
      if (unused_.empty()) {
        out = new float[kFrameSamples];
      } else {
        out = unused_.top();
        unused_.pop();
      }
    }
    std::memcpy(out, in, kFrameSamples * sizeof(float));
    std::lock_guard<std::mutex> g(mutex_);
    queued_.push(out);
  }

 private:
  void Fill(uint32_t i) {
    float* frame = nullptr;
    {
      std::lock_guard<std::mutex> g(mutex_);
      if (!queued_.empty()) {
        frame = queued_.front();
        queued_.pop();
      }
    }
    int16_t* dst = static_cast<int16_t*>(mem_[i]);
    if (!frame) {
      std::memset(dst, 0, kDataBytes);
      ++underruns_;
    } else {
      float tmp[kChannelSamples * 2];
      conversion::sequential_6_BE_to_interleaved_2_LE(tmp, frame, kChannelSamples,
                                                      GetStereoFold(), GetOutputGain());
      for (uint32_t k = 0; k < kChannelSamples * 2; ++k) {
        float v = tmp[k] * 32767.0f;
        dst[k] = static_cast<int16_t>(v > 32767.f ? 32767 : (v < -32768.f ? -32768 : v));
      }
      std::lock_guard<std::mutex> g(mutex_);
      unused_.push(frame);
    }
    wav_.Write(dst, kChannelSamples);
    armDCacheFlush(mem_[i], kBufferBytes);
    bufs_[i].data_size = kDataBytes;
    audoutAppendAudioOutBuffer(&bufs_[i]);
    semaphore_->Release(1, nullptr);
  }

  void Run() {
    { Result cm = svcSetThreadCoreMask(CUR_THREAD_HANDLE, -1, 0x7); if (oblivion_bootlog) oblivion_bootlog("audio: thread core mask rc=0x%x", (unsigned)cm); }
    for (uint32_t i = 0; i < kBufferCount; ++i) Fill(i);
    auto last_log = std::chrono::steady_clock::now();
    uint64_t frames = 0;
    while (running_) {
      AudioOutBuffer* released = nullptr;
      u32 count = 0;
      Result rc = audoutWaitPlayFinish(&released, &count, 50000000ull);
      if (R_FAILED(rc) || count == 0) continue;
      while (released) {
        uint32_t idx = static_cast<uint32_t>(released - bufs_);
        if (idx < kBufferCount) {
          Fill(idx);
          ++frames;
        }
        released = nullptr;
        count = 0;
        if (R_FAILED(audoutGetReleasedAudioOutBuffer(&released, &count)) || count == 0) break;
      }
      auto now = std::chrono::steady_clock::now();
      if (now - last_log > std::chrono::seconds(10)) {
        last_log = now;
        REXAPU_INFO("switch audio: {} frames played, {} underruns", frames, underruns_.load());
      }
    }
  }

  rex::thread::Semaphore* semaphore_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> underruns_{0};
  AudioOutBuffer bufs_[kBufferCount];
  void* mem_[kBufferCount] = {};
  std::mutex mutex_;
  std::queue<float*> queued_;
  std::stack<float*> unused_;
  WavWriter wav_;
};
}  // namespace

std::unique_ptr<AudioSystem> SwitchAudioSystem::Create(
    runtime::FunctionDispatcher* function_dispatcher) {
  return std::make_unique<SwitchAudioSystem>(function_dispatcher);
}

SwitchAudioSystem::SwitchAudioSystem(runtime::FunctionDispatcher* function_dispatcher)
    : AudioSystem(function_dispatcher) {}

SwitchAudioSystem::~SwitchAudioSystem() = default;

X_STATUS SwitchAudioSystem::CreateDriver(size_t /*index*/, rex::thread::Semaphore* semaphore,
                                         AudioDriver** out_driver) {
  auto* d = new AudoutDriver(memory_, semaphore);
  if (!d->Initialize()) {
    delete d;
    return X_STATUS_UNSUCCESSFUL;
  }
  *out_driver = d;
  return X_STATUS_SUCCESS;
}

void SwitchAudioSystem::DestroyDriver(AudioDriver* driver) {
  auto* d = static_cast<AudoutDriver*>(driver);
  d->Shutdown();
  delete d;
}

}  // namespace rex::audio::switch_hw

#endif
