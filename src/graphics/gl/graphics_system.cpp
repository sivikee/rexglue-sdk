#include <rex/graphics/gl/graphics_system.h>

#include "command_processor.h"

#include <rex/system/kernel_state.h>
#include <rex/ui/graphics_provider.h>

namespace rex::graphics::gl {

namespace {
// Placeholder provider: GL objects are created on the command processor thread (EGL contexts are thread-bound).
class GlProvider final : public ::rex::ui::GraphicsProvider {
 public:
  std::unique_ptr<::rex::ui::Presenter> CreatePresenter(
      ::rex::ui::Presenter::HostGpuLossCallback) override {
    return nullptr;
  }
  std::unique_ptr<::rex::ui::ImmediateDrawer> CreateImmediateDrawer() override { return nullptr; }
};
}  // namespace

GlGraphicsSystem::GlGraphicsSystem() = default;
GlGraphicsSystem::~GlGraphicsSystem() = default;

std::string GlGraphicsSystem::name() const {
  return "OpenGL";
}

void GlGraphicsSystem::CreateProvider(bool) {
  provider_ = std::make_unique<GlProvider>();
}

std::unique_ptr<CommandProcessor> GlGraphicsSystem::CreateCommandProcessor() {
  return std::make_unique<GlCommandProcessor>(this, kernel_state_);
}

}  // namespace rex::graphics::gl
