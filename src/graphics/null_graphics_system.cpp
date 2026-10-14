/**
 * @file        graphics/null_graphics_system.cpp
 * @brief       Headless "null Xenos" graphics system (see header).
 */
#include <rex/graphics/null_graphics_system.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>

#include <rex/chrono/clock.h>
#include <rex/graphics/xenos.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/system/function_dispatcher.h>
#include <rex/system/kernel_state.h>
#include <rex/thread.h>

namespace rex::graphics {

namespace {
constexpr uint32_t kMmioBase = 0x7FC80000;
constexpr uint32_t kRegCpRbWptr = 0x01C5;
constexpr uint32_t kRegScratchUmsk = 0x01DC;
constexpr uint32_t kRegScratchAddr = 0x01DD;
constexpr uint32_t kRegScratch0 = 0x0578;
constexpr uint32_t kRegCoherStatusHost = 0x0A31;
constexpr uint32_t kRegVgtEventInitiator = 0x21F9;
constexpr uint32_t kRegSampleCountAddr = 0x2325;
constexpr uint32_t kRegCount = 0x10000;

inline uint32_t bswap(uint32_t v) {
  return __builtin_bswap32(v);
}
}  // namespace

// Cursor over a dword ring (primary buffer, wraps) or a linear indirect buffer.
struct NullGraphicsSystem::Reader {
  const uint32_t* base;
  uint32_t size;  // dwords
  uint32_t rd;
  uint32_t wr;
  uint32_t remaining() const { return wr >= rd ? wr - rd : size - rd + wr; }
  uint32_t Next() {
    uint32_t v = bswap(base[rd]);
    if (++rd >= size) rd = 0;
    return v;
  }
};

NullGraphicsSystem::NullGraphicsSystem() : regs_(kRegCount, 0) {}
NullGraphicsSystem::~NullGraphicsSystem() = default;

X_STATUS NullGraphicsSystem::SetupGuestGpu(runtime::FunctionDispatcher* function_dispatcher,
                                           system::KernelState* kernel_state) {
  dispatcher_ = function_dispatcher;
  kernel_state_ = kernel_state;
  memory_ = function_dispatcher->memory();
  // GPU registers live at 0x7FC80000-0x7FC8FFFF. Generated code reaches them through
  // MMIOHandler::CheckLoad/CheckStore (REX_MM_*), which dispatches to these callbacks on every
  // platform (no page-fault handler needed).
  memory_->AddVirtualMappedRange(
      kMmioBase, 0xFFFF0000, 0x0000FFFF, this,
      reinterpret_cast<runtime::MMIOReadCallback>(ReadRegisterThunk),
      reinterpret_cast<runtime::MMIOWriteCallback>(WriteRegisterThunk));
  running_ = true;

  worker_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        WorkerMain();
        return 0;
      }));
  worker_->set_name("GPU Commands (null)");
  worker_->Create();

  vsync_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state_, 128 * 1024, 0, [this]() {
        VsyncMain();
        return 0;
      }));
  vsync_->set_name("GPU VSync (null)");
  vsync_->Create();
  REXGPU_INFO("NullGraphicsSystem: up (MMIO range 0x{:08X} registered)", kMmioBase);
  return X_STATUS_SUCCESS;
}

void NullGraphicsSystem::Shutdown() {
  running_ = false;
  if (worker_) {
    worker_->Wait(0, 0, 0, nullptr);
    worker_.reset();
  }
  if (vsync_) {
    vsync_->Wait(0, 0, 0, nullptr);
    vsync_.reset();
  }
}

uint32_t NullGraphicsSystem::ReadRegisterThunk(void*, NullGraphicsSystem* gs, uint32_t addr) {
  return gs->MmioRead((addr & 0xFFFF) / 4);
}

void NullGraphicsSystem::WriteRegisterThunk(void*, NullGraphicsSystem* gs, uint32_t addr,
                                            uint32_t value) {
  gs->MmioWrite((addr & 0xFFFF) / 4, value);
}

// Same register values the real GraphicsSystem::ReadRegister returns.
uint32_t NullGraphicsSystem::MmioRead(uint32_t r) {
  system::X_VIDEO_MODE mode;
  switch (r) {
    case 0x0F00:  // RB_EDRAM_TIMING
      return 0x08100748;
    case 0x0F01:  // RB_BC_CONTROL
      return 0x0000200E;
    case 0x194C:  // D1MODE_V_COUNTER
      kernel::xboxkrnl::VdQueryVideoMode(&mode);
      return std::min<uint32_t>(mode.display_height, 0x0FFF);
    case 0x1951:  // interrupt status: vblank
      return 1;
    case 0x1961:  // D1MODE_VIEWPORT_SIZE
      kernel::xboxkrnl::VdQueryVideoMode(&mode);
      return (std::min<uint32_t>(mode.display_width, 0x0FFF) << 16) |
             std::min<uint32_t>(mode.display_height, 0x0FFF);
    default:
      return regs_[r & 0xFFFF];
  }
}

void NullGraphicsSystem::MmioWrite(uint32_t r, uint32_t value) {
  regs_[r & 0xFFFF] = value;
  if (r == kRegCpRbWptr) {
    wptr_.store(value, std::memory_order_release);
  }
}

void NullGraphicsSystem::SetInterruptCallback(uint32_t callback, uint32_t user_data) {
  interrupt_data_ = user_data;
  interrupt_cb_ = callback;
  REXGPU_INFO("NullGraphicsSystem: SetInterruptCallback({:08X}, {:08X})", callback, user_data);
}

void NullGraphicsSystem::InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) {
  read_index_ = 0;
  ring_size_bytes_ = uint32_t(1) << (size_log2 + 3);
  ring_ptr_ = ptr;
  REXGPU_INFO("NullGraphicsSystem: ring buffer {:08X}, {} bytes", ptr, ring_size_bytes_.load());
}

void NullGraphicsSystem::EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) {
  rptr_writeback_ = ptr;
  REXGPU_INFO("NullGraphicsSystem: RPtr write-back at {:08X}", ptr);
}

void NullGraphicsSystem::Dispatch(uint32_t source, uint32_t cpu) {
  uint32_t cb = interrupt_cb_;
  if (!cb) return;
  auto* thread = system::XThread::GetCurrentThread();
  if (!thread) return;
  if (cpu == 0xFFFFFFFF) cpu = 2;
  uint64_t args[] = {source, interrupt_data_.load()};
  dispatcher_->ExecuteInterrupt(thread->thread_state(), cb, args, 2);
  interrupts_.fetch_add(1, std::memory_order_relaxed);
}

void NullGraphicsSystem::VsyncMain() {
  using clk = std::chrono::steady_clock;
  const auto interval = std::chrono::microseconds(16667);
  auto next = clk::now() + interval;
  auto report = clk::now();
  uint64_t last_swaps = 0, last_vblanks = 0;
  while (running_) {
    auto now = clk::now();
    if (now < next) {
      rex::thread::Sleep(std::chrono::microseconds(std::min<int64_t>(
          2000, std::chrono::duration_cast<std::chrono::microseconds>(next - now).count())));
      continue;
    }
    next += interval;
    if (next < now) next = now + interval;  // fell behind: do not burst
    vblanks_.fetch_add(1, std::memory_order_relaxed);
    ++counter_;
    Dispatch(0, 2);

    if (now - report >= std::chrono::seconds(5)) {
      double secs = std::chrono::duration<double>(now - report).count();
      uint64_t sw = swaps_, vb = vblanks_;
      REXGPU_INFO(
          "GPU heartbeat: swaps={} ({:.1f}/s) vblanks={} ({:.1f}/s) packets={} draws_skipped={} "
          "irq={} waits={} rd={} wr={}",
          sw, (sw - last_swaps) / secs, vb, (vb - last_vblanks) / secs, packets_.load(),
          draws_.load(), interrupts_.load(), waits_.load(), read_index_, wptr_.load());
      last_swaps = sw;
      last_vblanks = vb;
      report = now;
    }
  }
}

void NullGraphicsSystem::WorkerMain() {
  uint32_t idle = 0;
  while (running_) {
    uint32_t ptr = ring_ptr_, bytes = ring_size_bytes_;
    if (!ptr || !bytes) {
      rex::thread::Sleep(std::chrono::milliseconds(1));
      continue;
    }
    uint32_t size = bytes / 4;
    uint32_t wr = wptr_.load(std::memory_order_acquire);
    if (wr >= size || wr == read_index_) {
      if (++idle > 200) {
        rex::thread::Sleep(std::chrono::microseconds(500));
      } else {
        rex::thread::MaybeYield();
      }
      continue;
    }
    idle = 0;
    Reader r{reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(ptr)), size, read_index_,
             wr};
    Execute(r);
    read_index_ = wr;
    uint32_t wb = rptr_writeback_;
    if (wb) {
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(wb), read_index_);
      std::atomic_thread_fence(std::memory_order_release);
    }
  }
}

void NullGraphicsSystem::Execute(Reader& r) {
  while (running_ && r.remaining()) {
    if (!ExecutePacket(r)) {
      REXGPU_ERROR("NullGraphicsSystem: bad packet, dropping rest of buffer");
      r.rd = r.wr;
      break;
    }
    packets_.fetch_add(1, std::memory_order_relaxed);
  }
}

uint32_t NullGraphicsSystem::ReadRegister(uint32_t index) {
  return regs_[index & 0xFFFF];
}

void NullGraphicsSystem::WriteRegister(uint32_t index, uint32_t value) {
  index &= 0xFFFF;
  regs_[index] = value;
  if (index >= kRegScratch0 && index < kRegScratch0 + 8) {
    uint32_t n = index - kRegScratch0;
    if ((1u << n) & regs_[kRegScratchUmsk]) {
      uint32_t addr = regs_[kRegScratchAddr] + n * 4;
      memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(addr), value);
    }
  }
}

bool NullGraphicsSystem::ExecutePacket(Reader& r) {
  uint32_t packet = r.Next();
  if (packet == 0) return true;
  switch (packet >> 30) {
    case 0: {
      uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
      if (r.remaining() < count) return false;
      uint32_t base = packet & 0x7FFF;
      bool one = (packet >> 15) & 1;
      for (uint32_t i = 0; i < count; ++i) WriteRegister(one ? base : base + i, r.Next());
      return true;
    }
    case 1: {
      if (r.remaining() < 2) return false;
      uint32_t d1 = r.Next(), d2 = r.Next();
      WriteRegister(packet & 0x7FF, d1);
      WriteRegister((packet >> 11) & 0x7FF, d2);
      return true;
    }
    case 2:
      return true;
    default:
      return ExecuteType3(r, packet);
  }
}

static bool Compare(uint32_t func, uint32_t v, uint32_t mask, uint32_t ref) {
  v &= mask;
  switch (func & 7) {
    case 0:
      return false;
    case 1:
      return v < ref;
    case 2:
      return v <= ref;
    case 3:
      return v == ref;
    case 4:
      return v != ref;
    case 5:
      return v >= ref;
    case 6:
      return v > ref;
    default:
      return true;
  }
}

bool NullGraphicsSystem::ExecuteType3(Reader& r, uint32_t packet) {
  using namespace xenos;
  uint32_t opcode = (packet >> 8) & 0x7F;
  uint32_t count = ((packet >> 16) & 0x3FFF) + 1;
  if (r.remaining() < count) return false;
  uint32_t end = r.rd + count;
  if (end >= r.size) end -= r.size;

  switch (opcode) {
    case PM4_INTERRUPT: {
      uint32_t cpu_mask = r.Next();
      for (int n = 0; n < 6; ++n)
        if (cpu_mask & (1u << n)) Dispatch(1, n);
      break;
    }
    case PM4_XE_SWAP:
      swaps_.fetch_add(1, std::memory_order_relaxed);
      ++counter_;
      break;
    case PM4_INDIRECT_BUFFER:
    case PM4_INDIRECT_BUFFER_PFD: {
      uint32_t ptr = r.Next() & 0x1FFFFFFF;
      uint32_t len = r.Next() & 0xFFFFF;
      Reader ib{reinterpret_cast<const uint32_t*>(memory_->TranslatePhysical(ptr)), len + 1, 0,
                len};
      Execute(ib);
      break;
    }
    case PM4_WAIT_REG_MEM: {
      uint32_t info = r.Next(), addr = r.Next(), ref = r.Next(), mask = r.Next(), wait = r.Next();
      bool is_mem = (info & 0x10) != 0;
      waits_.fetch_add(1, std::memory_order_relaxed);
      int spins = 0;
      uint32_t total_spins = 0;
      while (running_) {
        uint32_t v;
        if (is_mem) {
          v = *reinterpret_cast<volatile uint32_t*>(memory_->TranslatePhysical(addr & ~3u));
          v = GpuSwap(v, static_cast<Endian>(addr & 3));
        } else {
          if (addr == kRegCoherStatusHost) regs_[kRegCoherStatusHost] = 0;  // always coherent
          v = ReadRegister(addr);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (Compare(info, v, mask, ref)) break;
        if (++total_spins == 2000) {
          REXGPU_WARN(
              "NullGraphicsSystem: WAIT_REG_MEM stuck: info={:08X} addr={:08X} ref={:08X} "
              "mask={:08X} wait={:X} value={:08X}",
              info, addr, ref, mask, wait, v);
        }
        if (wait >= 0x100 || ++spins > 100) {
          rex::thread::Sleep(std::chrono::milliseconds(1));
        } else {
          rex::thread::MaybeYield();
        }
      }
      break;
    }
    case PM4_REG_RMW: {
      uint32_t info = r.Next(), and_mask = r.Next(), or_mask = r.Next();
      uint32_t v = ReadRegister(info & 0x1FFF);
      v &= ((info >> 31) & 1) ? ReadRegister(and_mask & 0x1FFF) : and_mask;
      v |= ((info >> 30) & 1) ? ReadRegister(or_mask & 0x1FFF) : or_mask;
      WriteRegister(info & 0x1FFF, v);
      break;
    }
    case PM4_REG_TO_MEM: {
      uint32_t reg = r.Next(), mem = r.Next();
      uint32_t v = GpuSwap(ReadRegister(reg), static_cast<Endian>(mem & 3));
      *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(mem & ~3u)) = v;
      break;
    }
    case PM4_MEM_WRITE: {
      uint32_t addr = r.Next();
      for (uint32_t i = 0; i + 1 < count; ++i) {
        uint32_t d = GpuSwap(r.Next(), static_cast<Endian>(addr & 3));
        *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(addr & ~3u)) = d;
        addr += 4;
      }
      break;
    }
    case PM4_COND_WRITE: {
      uint32_t info = r.Next(), poll = r.Next(), ref = r.Next(), mask = r.Next();
      uint32_t waddr = r.Next(), wdata = r.Next();
      uint32_t v;
      if (info & 0x10) {
        v = GpuSwap(*reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(poll & ~3u)),
                    static_cast<Endian>(poll & 3));
      } else {
        v = ReadRegister(poll);
      }
      if (Compare(info, v, mask, ref)) {
        if (info & 0x100) {
          *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(waddr & ~3u)) =
              GpuSwap(wdata, static_cast<Endian>(waddr & 3));
        } else {
          WriteRegister(waddr, wdata);
        }
      }
      break;
    }
    case PM4_EVENT_WRITE: {
      WriteRegister(kRegVgtEventInitiator, r.Next() & 0x3F);
      break;
    }
    case PM4_EVENT_WRITE_SHD: {
      uint32_t init = r.Next(), addr = r.Next(), value = r.Next();
      WriteRegister(kRegVgtEventInitiator, init & 0x3F);
      uint32_t d = ((init >> 31) & 1) ? counter_ : value;
      d = GpuSwap(d, static_cast<Endian>(addr & 3));
      *reinterpret_cast<uint32_t*>(memory_->TranslatePhysical(addr & ~3u)) = d;
      std::atomic_thread_fence(std::memory_order_release);
      break;
    }
    case PM4_EVENT_WRITE_EXT: {
      uint32_t init = r.Next(), addr = r.Next();
      WriteRegister(kRegVgtEventInitiator, init & 0x3F);
      uint16_t ext[] = {0,
                        uint16_t(kTexture2DCubeMaxWidthHeight >> 3),
                        0,
                        uint16_t(kTexture2DCubeMaxWidthHeight >> 3),
                        0,
                        1};
      auto* dst = reinterpret_cast<uint16_t*>(memory_->TranslatePhysical(addr & ~3u));
      for (int i = 0; i < 6; ++i) dst[i] = __builtin_bswap16(ext[i]);
      break;
    }
    case PM4_EVENT_WRITE_ZPD: {
      const uint32_t kFinished = bswap(0xFFFFFEED);
      WriteRegister(kRegVgtEventInitiator, r.Next() & 0x3F);
      uint32_t a = regs_[kRegSampleCountAddr];
      if (a) {
        auto* s = memory_->TranslatePhysical<xe_gpu_depth_sample_counts*>(a);
        bool fin = s->ZPass_A == kFinished || s->ZPass_B == kFinished || s->ZFail_A == kFinished ||
                   s->ZFail_B == kFinished;
        std::memset(s, 0, sizeof(*s));
        if (fin) {
          s->ZPass_A = 1000;
          s->Total_A = 1000;
        }
      }
      break;
    }
    case PM4_DRAW_INDX:
    case PM4_DRAW_INDX_2:
    case PM4_DRAW_INDX_BIN:
    case PM4_DRAW_INDX_2_BIN:
      draws_.fetch_add(1, std::memory_order_relaxed);
      break;
    default:
      break;  // everything else is consumed without effect
  }
  r.rd = end;
  return true;
}

}  // namespace rex::graphics
