#pragma once
#include <cstdint>
#include <rex/cvar.h>

REXCVAR_DECLARE(bool, gl_stream_uploads);
REXCVAR_DECLARE(bool, gl_sync_resolve);
REXCVAR_DECLARE(bool, gl_resolve_gpu_copy);
REXCVAR_DECLARE(int32_t, gl_resolve_max_age);
REXCVAR_DECLARE(bool, gl_state_cache);
