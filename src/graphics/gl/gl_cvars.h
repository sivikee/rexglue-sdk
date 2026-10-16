#pragma once
#include <cstdint>
#include <rex/cvar.h>

REXCVAR_DECLARE(bool, gl_stream_uploads);
REXCVAR_DECLARE(bool, gl_sync_resolve);
REXCVAR_DECLARE(bool, gl_resolve_gpu_copy);
REXCVAR_DECLARE(int32_t, gl_resolve_max_age);
REXCVAR_DECLARE(bool, gl_state_cache);
REXCVAR_DECLARE(bool, gl_gpu_resolve);
REXCVAR_DECLARE(bool, gl_gpu_front);
REXCVAR_DECLARE(bool, gl_cpu_frontbuffer);
REXCVAR_DECLARE(int32_t, gl_resolve_log);
REXCVAR_DECLARE(bool, gl_gpu_timers);
REXCVAR_DECLARE(int32_t, gl_swap_interval);
REXCVAR_DECLARE(double, gl_gpu_timestamp_scale);
REXCVAR_DECLARE(bool, gl_lazy_resolve);
REXCVAR_DECLARE(bool, gl_present_snapshot);
