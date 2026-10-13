// STUB (Nintendo Switch port, phase 1): dynamic library loading.
// Homebrew has no dlopen of host shared objects. Load() always fails; GetRawSymbol()
// returns nullptr. Callers (RenderDoc, Vulkan loader, GPU plugin loader) must treat
// that as "unavailable". The GPU plugin must be linked statically on Switch.
#include <rex/platform.h>
#include <rex/platform/dynlib.h>

namespace rex::platform {

DynamicLibrary::~DynamicLibrary() {
  Close();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this != &other) {
    Close();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

bool DynamicLibrary::Load(const std::filesystem::path&, SymbolResolution) {
  return false;
}

void DynamicLibrary::Close() {
  handle_ = nullptr;
}

void* DynamicLibrary::GetRawSymbol(const char*) const {
  return nullptr;
}

}  // namespace rex::platform
