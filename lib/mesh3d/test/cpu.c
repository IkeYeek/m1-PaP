
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "mesh3d.h"

#define CPU_COLORS 28
static unsigned cpu_colors[CPU_COLORS] __attribute__ ((unused)) = {
    0xFFFF0080, // Yellow
    0xFF000080, // Red
    0x00FF0080, // Green
    0xAE4AFF80, // Purple
    0x00FFFF80, // Cyan
    0xB0B0B080, // Grey
    0x6464FF80, // Blue
    0xFFBFF780, // Pale Pink
    0xFFD59180, // Cream
    0xCFFFBF80, // Pale Green
    0xF0808080, // Light Coral
    0xE000E080, // Magenta
    0x4B944780, // Dark green
    0x964B0080, // Brown
    // Same colors, but highlighted
    0xFFFF00FF, // Yellow
    0xFF0000FF, // Red
    0x00FF00FF, // Green
    0xAE4AFFFF, // Purple
    0x00FFFFFF, // Cyan
    0xB0B0B0FF, // Grey
    0x6464FFFF, // Blue
    0xFFBFF7FF, // Pale Pink
    0xFFD591FF, // Cream
    0xCFFFBFFF, // Pale Green
    0xF08080FF, // Light Coral
    0xE000E0FF, // Magenta
    0x4B9447FF, // Dark green
    0x964B00FF, // Brown
};

// settings
const unsigned int SCR_WIDTH  = 1024;
const unsigned int SCR_HEIGHT = 768;

#define MAX_CTX 2

static mesh3d_obj_t mesh;
static mesh3d_ctx_t ctx[MAX_CTX] = {NULL, NULL};

static unsigned nb_ctx = 2;

static void do_pick (void)
{
  int p = mesh3d_perform_picking (ctx, nb_ctx);
  if (p != -1)
    mesh3d_set_cpu_color (ctx[1], p, 1, 0xFFFFFFFF);
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

static void load_mesh (int argc, char *argv[], mesh3d_obj_t *mesh)
{
  if (argc > 1) {
    unsigned group = 1;

    if (argc > 2)
      group = atoi (argv[2]);

    if (!strcmp (argv[1], "--help") || !strcmp (argv[1], "-h")) {
      printf ("Usage: %s [ --torus | --cube | --morton | <filename> ] "
              "[cell_size]\n",
              argv[0]);
      exit (0);
    } else if (!strcmp (argv[1], "--torus") || !strcmp (argv[1], "-t"))
      mesh3d_obj_build_torus_surface (mesh, group);
    else if (!strcmp (argv[1], "--cube") || !strcmp (argv[1], "-c"))
      mesh3d_obj_build_cube (mesh, group);
    else if (!strcmp (argv[1], "--cubus-volumus") || !strcmp (argv[1], "-cv"))
      mesh3d_obj_build_cube_volume (mesh, group == 1 ? 4 : group);
    else if (!strcmp (argv[1], "--torus-volumus") || !strcmp (argv[1], "-tv"))
      mesh3d_obj_build_torus_volume (mesh, 32, 16, 16);
    else
      mesh3d_obj_load (argv[1], mesh);
  } else
    mesh3d_obj_build_default (mesh);
}

int main (int argc, char *argv[])
{
  mesh3d_obj_init (&mesh);
  load_mesh (argc, argv, &mesh);

  mesh3d_init (NULL);

  // Create SDL windows and initialize OpenGL context
  ctx[0] =
      mesh3d_ctx_create ("Data view", SDL_WINDOWPOS_CENTERED,
                         SDL_WINDOWPOS_UNDEFINED, SCR_WIDTH, SCR_HEIGHT, MESH3D_ENABLE_CLIPPING);

  // Attach mesh
  mesh3d_set_mesh (ctx[0], &mesh);

  // Heat Palette (as simple as defining these five key colors)
  float colors[] = {0.0f, 0.0f, 1.0f, 1.0f,  // blue
                    0.0f, 1.0f, 1.0f, 1.0f,  // cyan
                    0.0f, 1.0f, 0.0f, 1.0f,  // green
                    1.0f, 1.0f, 0.0f, 1.0f,  // yellow
                    1.0f, 0.0f, 0.0f, 1.0f}; // red

  mesh3d_configure_data_colors (ctx[0], colors, 5);

  float values[mesh.nb_cells];

  for (int i = 0; i < mesh.nb_cells; i++) {
    int d     = i <= mesh.nb_cells / 2 ? i : mesh.nb_cells - i - 1;
    values[i] = (float)d / (float)(mesh.nb_cells / 2);
  }

  mesh3d_set_data_colors (ctx[0], values);

  ctx[1] = mesh3d_ctx_create ("CPU view", SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED, 512, 384,
                              MESH3D_ENABLE_PICKING | MESH3D_ENABLE_CLIPPING);

  mesh3d_set_mesh (ctx[1], &mesh);

  // Color cell according to CPU
  mesh3d_use_cpu_colors (ctx[1]);

  // Some initial colors…
  for (int c = 0; c < mesh.nb_cells; c++)
    mesh3d_set_cpu_color (ctx[1], c, 1, cpu_colors[c % 14 + 14]);

  // render loop
  while (1) {
    process_events ();
    mesh3d_render (ctx, nb_ctx);
  }

  return 0;
}
