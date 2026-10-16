# OpenGL 4.3 (EGL/Mesa nouveau) Xenos backend for the Nintendo Switch. Static library linked by the app
# (together with -lglad -lEGL -lglapi -ldrm_nouveau, see ports/switch/oblivion_gl/build.sh).
set(REXGPU_GL_SOURCES
    # Backend-agnostic Xenos core
    flags.cpp
    register_file.cpp
    registers.cpp
    pipeline/shader/shader.cpp
    format/ucode.cpp
    sampler_info.cpp
    graphics_system.cpp
    command_processor.cpp
    util/draw_extent_estimator.cpp
    util/draw.cpp
    packet_disassembler.cpp
    primitive_processor.cpp
    shared_memory.cpp
    pipeline/shader/interpreter.cpp
    pipeline/shader/translator.cpp
    pipeline/shader/translator_disasm.cpp

    # SPIR-V shader translation (Xenos ucode -> SPIR-V -> SPIRV-Cross -> GLSL 4.30)
    pipeline/shader/spirv_builder.cpp
    pipeline/shader/spirv.cpp
    pipeline/shader/spirv_translator.cpp
    pipeline/shader/spirv_translator_alu.cpp
    pipeline/shader/spirv_translator_fetch.cpp
    pipeline/shader/spirv_translator_memexport.cpp
    pipeline/shader/spirv_translator_rb.cpp

    # GL backend
    gl/gl_context.cpp
    gl/graphics_system.cpp
    gl/command_processor.cpp
    gl/perf.cpp
    gl/spirv_to_glsl.cpp
    gl/shader.cpp
    gl/shared_memory.cpp
    gl/primitive_processor.cpp
    gl/pipeline_cache.cpp
    gl/render_targets.cpp
    gl/texture_cache.cpp
)
list(TRANSFORM REXGPU_GL_SOURCES PREPEND ${CMAKE_CURRENT_LIST_DIR}/../)
add_library(rexgpu-gl STATIC ${REXGPU_GL_SOURCES})
target_include_directories(rexgpu-gl PRIVATE
    ${REXGLUE_ROOT}/thirdparty
    ${REXGLUE_ROOT}/thirdparty/SPIRV-Cross
    $ENV{DEVKITPRO}/portlibs/switch/include
    $ENV{DEVKITPRO}/libnx/include)
target_compile_options(rexgpu-gl PRIVATE -Wno-unused-variable)
target_link_libraries(rexgpu-gl PUBLIC rexruntime xxHash::xxhash glslang::SPIRV spirv-cross-glsl spirv-cross-core)
