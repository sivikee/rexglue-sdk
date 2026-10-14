/**
 * @file        graphics/gl/graphics_system.h
 * @brief       OpenGL 4.3 (Mesa/nouveau via EGL) Xenos graphics system for the Switch port.
 *
 * Derives from the real rex::graphics::GraphicsSystem (ring buffer, MMIO, interrupts, vblank worker)
 * and uses a GL command processor. No ui::Presenter: the command processor presents from IssueSwap.
 */
#pragma once

#include <rex/graphics/graphics_system.h>

namespace rex::graphics::gl {

class GlGraphicsSystem final : public GraphicsSystem {
 public:
  GlGraphicsSystem();
  ~GlGraphicsSystem() override;

  std::string name() const override;

  // The headless runtime path only calls SetupGuestGpu; presentation is done by the command processor.
  X_STATUS SetupPresentation(::rex::ui::WindowedAppContext*) override { return X_STATUS_SUCCESS; }

 protected:
  void CreateProvider(bool with_presentation) override;
  std::unique_ptr<CommandProcessor> CreateCommandProcessor() override;
};

}  // namespace rex::graphics::gl
