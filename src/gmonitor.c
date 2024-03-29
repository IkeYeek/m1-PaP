#include "gmonitor.h"
#include "cpustat.h"
#include "debug.h"
#include "error.h"
#include "global.h"
#include "graphics.h"
#include "img_data.h"
#include "mesh3d_sdl_gl.h"
#include "mesh_data.h"
#include "monitoring.h"
#include "trace_common.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define LOAD_INTENSITY

#ifdef LOAD_INTENSITY

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

#endif

unsigned do_gmonitor = 0;

static const unsigned MONITOR_WIDTH  = 512;
static const unsigned MONITOR_HEIGHT = 512;

static SDL_Window *win      = NULL;
static SDL_Renderer *ren    = NULL;
static SDL_Texture *texture = NULL;

static Uint32 *restrict trace_img = NULL;

void gmonitor_init (int x, int y)
{
  PRINT_DEBUG ('g', "Gmonitor window: %u x %u\n", MONITOR_WIDTH,
               MONITOR_HEIGHT);

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    // Création de la fenêtre sur l'écran
    win = SDL_CreateWindow ("Tile Mapping", x, y, MONITOR_WIDTH, MONITOR_HEIGHT,
                            SDL_WINDOW_SHOWN);
    if (win == NULL)
      exit_with_error ("SDL_CreateWindow failed (%s)", SDL_GetError ());

    // Initialisation du moteur de rendu
    ren = SDL_CreateRenderer (
        win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (ren == NULL)
      exit_with_error ("SDL_CreateRenderer failed (%s)", SDL_GetError ());

    SDL_RendererInfo info;
    SDL_GetRendererInfo (ren, &info);
    PRINT_DEBUG ('g', "Tiling window renderer: [%s]\n", info.name);

    // Creation d'une surface capable de mémoire quel processeur/thread a
    // travaillé sur quel pixel
    trace_img = malloc (DIM * DIM * sizeof (Uint32));
    bzero (trace_img, DIM * DIM * sizeof (Uint32));

    // Création d'une texture DIM x DIM sur la carte graphique
    texture = SDL_CreateTexture (
        ren, SDL_PIXELFORMAT_RGBA8888, // SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STATIC, DIM, DIM);
    if (texture == NULL)
      exit_with_error ("SDL_CreateTexture failed (%s)", SDL_GetError ());

    SDL_SetTextureBlendMode (texture, SDL_BLENDMODE_BLEND);

  } else {
    ctx[nb_ctx++] =
        mesh3d_ctx_create ("Tile Mapping", x, y, 768, 576,
                           MESH3D_ENABLE_CLIPPING);
    win = mesh3d_sdl_window (ctx[1]);
    mesh3d_set_mesh (ctx[1], &mesh);

    // Color cell according to CPU
    mesh3d_use_cpu_colors (ctx[1]);
  }

  if (TTF_Init () < 0)
    exit_with_error ("TTF_Init failed (%s)", TTF_GetError ());

  {
    int x = -1, y = -1, h = -1, w = -1;

    SDL_GetWindowPosition (win, &x, &y);
    SDL_GetWindowSize (win, &w, &h);

    cpustat_init (x + w, y);
  }
}

void __gmonitor_start_iteration (long time)
{
  cpustat_reset (time);

#ifdef LOAD_INTENSITY
  prev_max_duration = max_duration;
  max_duration      = 0;
#endif

  if (easypap_mode == EASYPAP_MODE_3D_MESHES)
    mesh3d_reset_cpu_colors (ctx[1]);
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
    unsigned color = cpu_colors[who % MAX_COLORS];

    t1 = what_time_is_it ();

#ifdef LOAD_INTENSITY
    if (duration > max_duration)
      max_duration = duration;

    if (heat_mode && prev_max_duration) { // not the first iteration
      if (duration <= prev_max_duration) {
        float r, g, b, f;
        const float scale = 1.0 / 14.0;
        long intensity    = 8191 * duration / prev_max_duration;
        // log2(intensity) is in [0..12]
        // so log2(intensity) + 2 is in [2..14]
        f     = (float)(mylog2 (intensity) + 2) * scale;
        r     = (color >> 24) * f;
        g     = ((color >> 16) & 255) * f;
        b     = ((color >> 8) & 255) * f;
        color = ((int)r) << 24 | ((int)g) << 16 | ((int)b) << 8 | 255;
      }
    }
#endif

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
      for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
          trace_img[i * DIM + j] = color;
    } else {
      // 3D Mesh
      mesh3d_set_cpu_color (ctx[1], x, width, color);
    }
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

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    SDL_Rect src, dst;

    SDL_UpdateTexture (texture, NULL, trace_img, DIM * sizeof (Uint32));

    src.x = 0;
    src.y = 0;
    src.w = DIM;
    src.h = DIM;

    // On redimensionne l'image pour qu'elle occupe toute la fenêtre
    dst.x = 0;
    dst.y = 0;
    dst.w = MONITOR_WIDTH;
    dst.h = MONITOR_HEIGHT;

    SDL_RenderClear (ren);

    SDL_RenderCopy (ren, texture, &src, &dst);

    SDL_RenderPresent (ren);

    bzero (trace_img, DIM * DIM * sizeof (Uint32));
  }
  // Note: nothing todo if mesh3d is used
}

void gmonitor_clean ()
{
  if (ren != NULL)
    SDL_DestroyRenderer (ren);
  else
    return;

  if (win != NULL)
    SDL_DestroyWindow (win);
  else
    return;

  if (trace_img != NULL)
    free (trace_img);

  if (texture != NULL)
    SDL_DestroyTexture (texture);

  cpustat_clean ();

  TTF_Quit ();
}

void gmonitor_toggle_heat_mode (void)
{
#ifdef LOAD_INTENSITY
  if (do_gmonitor) {
    heat_mode ^= 1;
    printf ("< Heatmap mode %s >\n", heat_mode ? "ON" : "OFF");
  }
#endif
}
