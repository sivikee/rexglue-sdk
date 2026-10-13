/**
 * @file        graphics/null_graphics_system.h
 * @brief       "Null Xenos": a headless IGraphicsSystem that consumes the guest ring buffer
 *              without rendering (no draws, resolves or texture work).
 *
 * Self-contained: no ui::GraphicsProvider/Presenter, no plugin loader, no D3D12/Vulkan. It
 * parses PM4 packets only as far as the guest can observe: read-pointer write-back, fences
 * (EVENT_WRITE_SHD, MEM_WRITE, COND_WRITE, WAIT_REG_MEM), interrupts, swaps (counted), scratch
 * register write-back. A 60 Hz vblank thread fires the guest graphics interrupt callback.
 * Used by the Switch port; nothing on other platforms references it.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include <rex/system/interfaces/graphics.h>
#include <rex/system/xthread.h>

namespace rex::memory {
class Memory;
}

namespace rex::graphics {

class NullGraphicsSystem : public system::IGraphicsSystem {
 public:
  NullGraphicsSystem();
  ~NullGraphicsSystem() override;

  X_STATUS SetupPresentation(ui::WindowedAppContext*) override { return X_STATUS_SUCCESS; }
  X_STATUS SetupGuestGpu(runtime::FunctionDispatcher* function_dispatcher,
                         system::KernelState* kernel_state) override;
  bool has_presentation() const override { return false; }

  void SetInterruptCallback(uint32_t callback, uint32_t user_data) override;
  void InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) override;
  void EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) override;
  void Shutdown() override;

  uint64_t swap_count() const { return swaps_.load(std::memory_order_relaxed); }
  uint64_t vblank_count() const { return vblanks_.load(std::memory_order_relaxed); }

 private:
  struct Reader;
  void WorkerMain();
  void VsyncMain();
  static uint32_t ReadRegisterThunk(void* ppc_context, NullGraphicsSystem* gs, uint32_t addr);
  static void WriteRegisterThunk(void* ppc_context, NullGraphicsSystem* gs, uint32_t addr,
                                 uint32_t value);
  uint32_t MmioRead(uint32_t reg);
  void MmioWrite(uint32_t reg, uint32_t value);
  void Execute(Reader& r);
  bool ExecutePacket(Reader& r);
  bool ExecuteType3(Reader& r, uint32_t packet);
  void WriteRegister(uint32_t index, uint32_t value);
  uint32_t ReadRegister(uint32_t index);
  void Dispatch(uint32_t source, uint32_t cpu);

  runtime::FunctionDispatcher* dispatcher_ = nullptr;
  system::KernelState* kernel_state_ = nullptr;
  memory::Memory* memory_ = nullptr;
  std::atomic<uint32_t> wptr_{0};  // CP_RB_WPTR, written by the guest through MMIO

  std::atomic<bool> running_{false};
  system::object_ref<system::XHostThread> worker_;
  system::object_ref<system::XHostThread> vsync_;

  std::atomic<uint32_t> ring_ptr_{0};
  std::atomic<uint32_t> ring_size_bytes_{0};
  std::atomic<uint32_t> rptr_writeback_{0};
  std::atomic<uint32_t> interrupt_cb_{0};
  std::atomic<uint32_t> interrupt_data_{0};
  uint32_t read_index_ = 0;  // dwords

  std::vector<uint32_t> regs_;
  uint32_t counter_ = 0;

  std::atomic<uint64_t> swaps_{0}, vblanks_{0}, packets_{0}, draws_{0}, interrupts_{0},
      waits_{0};
};

}  // namespace rex::graphics
