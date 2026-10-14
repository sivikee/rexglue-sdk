# OpenGL 4.3 (EGL/Mesa nouveau) Xenos backend for the Nintendo Switch. Static library linked by the app
# (together with -lglad -lEGL -lglapi -ldrm_nouveau, see ports/switch/oblivion_gl/build.sh).
set(REXGPU_GL_SOURCES
    # Backend-agnostic Xenos core
    flags.cpp
    register_file.cpp
    registers.cpp
    packet_disassembler.cpp
    graphics_system.cpp
    command_processor.cpp

    # GL backend
    gl/gl_context.cpp
    gl/graphics_system.cpp
    gl/command_processor.cpp
)
list(TRANSFORM REXGPU_GL_SOURCES PREPEND ${CMAKE_CURRENT_LIST_DIR}/../)
add_library(rexgpu-gl STATIC ${REXGPU_GL_SOURCES})
target_include_directories(rexgpu-gl PRIVATE
    ${REXGLUE_ROOT}/thirdparty
    $ENV{DEVKITPRO}/portlibs/switch/include
    $ENV{DEVKITPRO}/libnx/include)
target_compile_options(rexgpu-gl PRIVATE -Wno-unused-variable)
target_link_libraries(rexgpu-gl PUBLIC rexruntime xxHash::xxhash)
