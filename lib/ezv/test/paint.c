
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "mesh3d.h"

// settings
const unsigned int SCR_WIDTH  = 1024;
const unsigned int SCR_HEIGHT = 768;

#define MAX_CTX 2

static mesh3d_obj_t mesh;
static mesh3d_ctx_t ctx[MAX_CTX] = {NULL, NULL};
static unsigned nb_ctx           = 1;
static int hud                   = -1;

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
  int p = mesh3d_perform_picking (ctx, nb_ctx);

  if (p != -1) {
    data[p] = pencil_color; // selected triangle
    mesh3d_hud_set (ctx[0], hud, "Painted: %d", painted++);

    pencil_color = pencil_color + 0.002;
    if (pencil_color > 1.0)
      pencil_color = 0.1;

    mesh3d_set_data_colors (ctx[0], data);
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
    mesh3d_process_event (ctx, nb_ctx, &event, NULL, &pick);
    if (pick)
      do_pick ();
  }
}

int main (int argc, char *argv[])
{
  mesh3d_obj_init (&mesh);

  if (argc > 1)
    mesh3d_obj_load (argv[1], &mesh);
  else
    mesh3d_obj_build_torus_surface (&mesh, 1);

  mesh3d_init (NULL);

  // Create SDL windows and initialize OpenGL context
  ctx[0] = mesh3d_ctx_create ("Paint", SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_UNDEFINED, SCR_WIDTH, SCR_HEIGHT,
                              MESH3D_ENABLE_PICKING | MESH3D_ENABLE_HUD |
                                  MESH3D_ENABLE_CLIPPING);
  hud    = mesh3d_hud_alloc (ctx[0]);
  mesh3d_hud_toggle (ctx[0], hud);

  // Attach mesh
  mesh3d_set_mesh (ctx[0], &mesh);

  // Heat Palette (as simple as defining these five key colors)
  float colors[] = {0.0f, 0.0f, 1.0f, 1.0f,  // blue
                    0.0f, 1.0f, 1.0f, 1.0f,  // cyan
                    0.0f, 1.0f, 0.0f, 1.0f,  // green
                    1.0f, 1.0f, 0.0f, 1.0f,  // yellow
                    1.0f, 0.0f, 0.0f, 1.0f,  // red
                    1.0f, 0.0f, 1.0f, 1.0f}; // purple

  mesh3d_configure_data_colors (ctx[0], colors, 6);

  data = malloc (mesh.nb_cells * sizeof (float));

  for (int i = 0; i < mesh.nb_cells; i++)
    data[i] = 0.0;

  mesh3d_set_data_colors (ctx[0], data);

  // render loop
  while (1) {
    process_events ();
    mesh3d_render (ctx, nb_ctx);
  }

  return 0;
}
