#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "gmonitor.h"
#include "cpustat.h"
#include "debug.h"
#include "error.h"
#include "ezp_colors.h"
#include "ezv.h"
#include "global.h"
#include "img_data.h"
#include "mesh_data.h"
#include "monitoring.h"

static long prev_max_duration = 0;
static long max_duration      = 0;
static unsigned heat_mode     = 0;

static const char LogTable256[256] = {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    0,      0,      1,      1,      2,      2,      2,      2,
    3,      3,      3,      3,      3,      3,      3,      3,
    LT (4), LT (5), LT (5), LT (6), LT (6), LT (6), LT (6), LT (7),
    LT (7), LT (7), LT (7), LT (7), LT (7), LT (7), LT (7)};

static inline unsigned mylog2 (unsigned v)
{
  unsigned r; // r will be lg(v)
  unsigned int t;

  if ((t = (v >> 16)))
    r = 16 + LogTable256[t];
  else if ((t = (v >> 8)))
    r = 8 + LogTable256[t];
  else
    r = LogTable256[v];

  return r;
}

unsigned do_gmonitor = 0;

void gmonitor_init (void)
{
  int xn = -1, yn = -1;
  ezv_ctx_type_t ctx_type = (easypap_mode == EASYPAP_MODE_2D_IMAGES)
                      ? EZV_CTX_TYPE_IMG2D
                      : EZV_CTX_TYPE_MESH3D;

  ezp_ctx_create (ctx_type);

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
    ezv_img2d_set_img (ctx[1], &easypap_img_desc);
  else
    ezv_mesh3d_set_mesh (ctx[1], &easypap_mesh_desc);

  // Tile mapping window uses RGBA cpu colors
  ezv_use_cpu_colors (ctx[1]);

  ezp_ctx_coord_next (ctx_type, 2, &xn, &yn);

  cpustat_init (xn, yn);
}

void __gmonitor_start_iteration (long time)
{
  cpustat_reset (time);

  prev_max_duration = max_duration;
  max_duration      = 0;

  ezv_reset_cpu_colors (ctx[1]);
}

void __gmonitor_start_tile (long time, int who)
{
  cpustat_start_work (time, who);
}

void __gmonitor_end_tile (long time, int who, int x, int y, int width,
                          int height)
{
  long duration __attribute__ ((unused)) = cpustat_finish_work (time, who);

  if (width) { // task has an associated tile
    long t1, t2;
    uint32_t color = ezp_cpu_colors[who % EZP_MAX_COLORS];

    t1 = what_time_is_it ();

    if (duration > max_duration)
      max_duration = duration;

    if (heat_mode && prev_max_duration) { // not the first iteration
      if (duration <= prev_max_duration) {
        float f;
        const float scale = 1.0 / 14.0;
        long intensity    = 8191 * duration / prev_max_duration;
        uint8_t r, g, b;
        // log2(intensity) is in [0..12]
        // so log2(intensity) + 2 is in [2..14]
        f     = (float)(mylog2 (intensity) + 2) * scale;
        r     = ezv_c2r (color) * f;
        g     = ezv_c2g (color) * f;
        b     = ezv_c2b (color) * f;
        color = ezv_rgb (r, g, b);
      }
    }

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
      // 2D Img
      ezv_set_cpu_color_2D (ctx[1], x, width, y, height, color);
    else
      // 3D Mesh
      ezv_set_cpu_color_1D (ctx[1], x, width, color);

    t2 = what_time_is_it ();

    cpustat_deduct_idle (t2 - t1, who);
  }
}

void __gmonitor_tile (long start_time, long end_time, int who, int x, int y,
                      int width, int height)
{
  __gmonitor_start_tile (start_time, who);
  __gmonitor_end_tile (end_time, who, x, y, width, height);
}

void __gmonitor_end_iteration (long time)
{
  cpustat_freeze (time);

  cpustat_display_stats ();
}

void gmonitor_clean (void)
{
  cpustat_clean ();
}

void gmonitor_toggle_heat_mode (void)
{
  if (do_gmonitor) {
    heat_mode ^= 1;
    printf ("< Heatmap mode %s >\n", heat_mode ? "ON" : "OFF");
  }
}
