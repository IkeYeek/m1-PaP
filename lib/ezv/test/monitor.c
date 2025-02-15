
#include <SDL2/SDL.h>
#include <cglm/cglm.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "error.h"
#include "ezv.h"
#include "ezv_event.h"

// settings
#define NB_CPU 22
#define NB_GPU 2

unsigned int SCR_WIDTH  = 512;
unsigned int SCR_HEIGHT = 512;

#define MAX_CTX 1

static mon_obj_t monitor;
static ezv_ctx_t ctx[MAX_CTX] = {NULL};
static unsigned nb_ctx        = 1;

static void process_events (void)
{
  SDL_Event event;

  int r = ezv_get_event (&event, 0);

  if (r > 0)
    ezv_process_event (ctx, nb_ctx, &event, NULL, NULL);
}

#define delta 64.0

static float values[NB_CPU + NB_GPU];

static void set_values (void)
{
  static float phase = 0;

  for (int i = 0; i < NB_CPU + NB_GPU; i++)
    values[i] =
        (1.0 + sin (phase + (float)i * M_PI / (float)(NB_CPU + NB_GPU - 1))) /
        2.0;

  phase = fmod (phase + M_PI / delta, 2 * M_PI);

  ezv_set_data_colors (ctx[0], values);
}

static void set_cpu_colors (ezv_ctx_t ctx, ezv_palette_name_t name, int cyclic)
{
  ezv_palette_t palette;

  ezv_palette_init (&palette);
  ezv_palette_set_predefined (&palette, name);

  ezv_use_cpu_colors (ctx);
  for (int c = 0; c < NB_CPU + NB_GPU; c++)
    if (cyclic)
      ezv_set_cpu_color_1D (ctx, c, 1, ezv_palette_get_color_from_index (&palette, c));
    else
      ezv_set_cpu_color_1D (ctx, c, 1, ezv_palette_get_color_from_value (&palette, (float)c / (float)(NB_CPU + NB_GPU - 1)));
}

int main (int argc, char *argv[])
{
  ezv_init (NULL);

  mon_obj_init (&monitor, NB_CPU, NB_GPU);
  ezv_mon_get_suggested_window_size (&monitor, &SCR_WIDTH, &SCR_HEIGHT);

  // Create SDL windows and initialize OpenGL context
  ctx[0] = ezv_ctx_create (EZV_CTX_TYPE_MONITOR, "Monitoring",
                           SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_UNDEFINED,
                           SCR_WIDTH, SCR_HEIGHT, EZV_ENABLE_VSYNC);

  // Attach monitor info
  ezv_mon_set_moninfo (ctx[0], &monitor);

  set_cpu_colors (ctx[0], EZV_PALETTE_RAINBOW, 0);

  // render loop
  while (1) {
    // Set random values for CPU perfmeters
    set_values ();

    process_events ();
    ezv_render (ctx, nb_ctx);
  }

  return 0;
}
