/**
 * @file        graphics/gl/gl_ext.h
 * @brief       GL entry points beyond the portlibs glad (core 4.3, no extensions) loaded through
 *              eglGetProcAddress: ARB_buffer_storage, ARB_clip_control, ARB_texture_barrier, ...
 */
#pragma once

#include <glad/glad.h>

#ifndef GL_DYNAMIC_STORAGE_BIT
#define GL_DYNAMIC_STORAGE_BIT 0x0100
#define GL_MAP_PERSISTENT_BIT 0x0040
#define GL_MAP_COHERENT_BIT 0x0080
#endif
#ifndef GL_CLIP_ORIGIN
#define GL_CLIP_ORIGIN 0x935C
#define GL_CLIP_DEPTH_MODE 0x935D
#define GL_LOWER_LEFT 0x8CA1
#define GL_UPPER_LEFT 0x8CA2
#define GL_NEGATIVE_ONE_TO_ONE 0x935E
#define GL_ZERO_TO_ONE 0x935F
#endif
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif
#ifndef GL_COMPRESSED_RGB_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3
#endif
#ifndef GL_MIRROR_CLAMP_TO_EDGE
#define GL_MIRROR_CLAMP_TO_EDGE 0x8743
#endif
#ifndef GL_COMPRESSED_RG_RGTC2
#define GL_COMPRESSED_RED_RGTC1 0x8DBB
#define GL_COMPRESSED_RG_RGTC2 0x8DBD
#endif

namespace rex::graphics::gl {

extern void (*glBufferStorage_)(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags);
extern void (*glClipControl_)(GLenum origin, GLenum depth);
extern void (*glTextureBarrier_)();
extern void (*glPolygonOffsetClamp_)(GLfloat factor, GLfloat units, GLfloat clamp);

void LoadGlExtensions(void* (*get_proc_address)(const char*));

}  // namespace rex::graphics::gl
