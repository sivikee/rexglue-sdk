#pragma once
// Inline copy of texture_util::GetTiledOffset2D (the SDK one is out of line and called per texel).
// pitch_aligned must already be aligned to 32.
#include <cstdint>

namespace rex::graphics::gl {

inline uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch_aligned, uint32_t bytes_per_block_log2) {
  int32_t macro = int32_t((x >> 5) + (y >> 5) * (pitch_aligned >> 5)) << (bytes_per_block_log2 + 7);
  int32_t micro = int32_t((x & 7) + ((y & 0xE) << 2)) << bytes_per_block_log2;
  int32_t offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + int32_t((y & 1) << 4);
  return uint32_t(((offset & ~0x1FF) << 3) + int32_t((y & 16) << 7) + ((offset & 0x1C0) << 2) +
                  (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F));
}

}  // namespace rex::graphics::gl
