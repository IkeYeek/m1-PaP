
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "ezv.h"
#include "ezv_event.h"

// settings
const unsigned int SCR_WIDTH  = 1024;
const unsigned int SCR_HEIGHT = 768;

#define MAX_CTX 2

static mesh3d_obj_t mesh;
static ezv_ctx_t ctx[MAX_CTX] = {NULL, NULL};
static unsigned nb_ctx        = 1;
static int hud                = -1;

static float *data        = NULL;
static float pencil_color = 0.05;
static int painted        = 0;

static inline unsigned *cell_neighbors (int cell)
{
  unsigned i = mesh.index_first_neighbor[cell];
  return mesh.neighbors + i;
}

static void do_pick (void)
{
  int p = ezv_perform_1D_picking (ctx, nb_ctx);

  if (p != -1) {
    data[p] = pencil_color; // selected triangle
    ezv_hud_set (ctx[0], hud, "Painted: %d", painted++);

    pencil_color = pencil_color + 0.002;
    if (pencil_color > 1.0)
      pencil_color = 0.1;

    ezv_set_data_colors (ctx[0], data);
  }
}

static inline int get_event (SDL_Event *event, int blocking)
{
  return blocking ? SDL_WaitEvent (event) : SDL_PollEvent (event);
}

static unsigned skipped_events = 0;

static int clever_get_event (SDL_Event *event, int blocking)
{
  int r;
  static int prefetched = 0;
  static SDL_Event pr_event; // prefetched event

  if (prefetched) {
    *event     = pr_event;
    prefetched = 0;
    return 1;
  }

  r = get_event (event, blocking);

  if (r != 1)
    return r;

  // check if successive, similar events can be dropped
  if (event->type == SDL_MOUSEMOTION) {

    do {
      int ret_code = get_event (&pr_event, 0);
      if (ret_code == 1) {
        if (pr_event.type == SDL_MOUSEMOTION) {
          *event     = pr_event;
          prefetched = 0;
          skipped_events++;
        } else {
          prefetched = 1;
        }
      } else
        return 1;
    } while (prefetched == 0);
  }

  return 1;
}

static void process_events (void)
{
  SDL_Event event;

  int r = clever_get_event (&event, 1);

  if (r > 0) {
    int pick;
    ezv_process_event (ctx, nb_ctx, &event, NULL, &pick);
    if (pick)
      do_pick ();
  }
}

int main (int argc, char *argv[])
{
  ezv_init (NULL);

  mesh3d_obj_init (&mesh);

  if (argc > 1)
    mesh3d_obj_load (argv[1], &mesh);
  else
    mesh3d_obj_build_torus_surface (&mesh, 1);

  // Create SDL windows and initialize OpenGL context
  ctx[0] = ezv_ctx_create (EZV_CTX_TYPE_MESH3D, "Paint", SDL_WINDOWPOS_CENTERED,
                           SDL_WINDOWPOS_UNDEFINED, SCR_WIDTH, SCR_HEIGHT,
                           EZV_ENABLE_PICKING | EZV_ENABLE_HUD |
                               EZV_ENABLE_CLIPPING);
  hud    = ezv_hud_alloc (ctx[0]);
  ezv_hud_toggle (ctx[0], hud);

  // Attach mesh
  ezv_mesh3d_set_mesh (ctx[0], &mesh);

  // Heat Palette (as simple as defining these five key colors)
  float colors[] = {0.0f, 0.0f, 1.0f, 1.0f,  // blue
                    0.0f, 1.0f, 1.0f, 1.0f,  // cyan
                    0.0f, 1.0f, 0.0f, 1.0f,  // green
                    1.0f, 1.0f, 0.0f, 1.0f,  // yellow
                    1.0f, 0.0f, 0.0f, 1.0f,  // red
                    1.0f, 0.0f, 1.0f, 1.0f}; // purple

  ezv_use_data_colors (ctx[0], colors, 6);

  data = malloc (mesh.nb_cells * sizeof (float));

  for (int i = 0; i < mesh.nb_cells; i++)
    data[i] = 0.0;

  ezv_set_data_colors (ctx[0], data);

  // render loop
  while (1) {
    process_events ();
    ezv_render (ctx, nb_ctx);
  }

  return 0;
}
