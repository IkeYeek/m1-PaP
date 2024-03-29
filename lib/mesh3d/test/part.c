
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

static void do_pick (void)
{
  int p = mesh3d_perform_picking (ctx, nb_ctx);
  if (p == -1)
    mesh3d_hud_set (ctx[0], hud, NULL);
  else
    mesh3d_hud_set (ctx[0], hud, "Part: %d", mesh3d_obj_get_patch_of_cell(&mesh, p));
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

static int nb_patches = -1;

static void load_mesh (int argc, char *argv[], mesh3d_obj_t *mesh)
{
  if (argc > 1) {

    if (argc > 2)
      nb_patches = atoi (argv[2]);

    if (!strcmp (argv[1], "--help") || !strcmp (argv[1], "-h")) {
      printf ("Usage: %s [ -t1 | -t2 | -tv | -cv | -cy | <file> ] [#patches]\n",
              argv[0]);
      exit (0);
    } else if (!strcmp (argv[1], "--torus-1") || !strcmp (argv[1], "-t1"))
      mesh3d_obj_build_torus_surface (mesh, 1);
    else if (!strcmp (argv[1], "--torus-2") || !strcmp (argv[1], "-t2"))
      mesh3d_obj_build_torus_surface (mesh, 2);
    else if (!strcmp (argv[1], "--torus-volumus") || !strcmp (argv[1], "-tv"))
      mesh3d_obj_build_torus_volume (mesh, 32, 16, 16);
    else if (!strcmp (argv[1], "--cubus-volumus") || !strcmp (argv[1], "-cv"))
      mesh3d_obj_build_cube_volume (mesh, 16);
    else if (!strcmp (argv[1], "--cylinder-volumus") ||
             !strcmp (argv[1], "-cy"))
      mesh3d_obj_build_cylinder_volume (mesh, 400, 200);
    else if (!strcmp (argv[1], "--wall") ||
             !strcmp (argv[1], "-w"))
      mesh3d_obj_build_wall (mesh, 8);
    else
      mesh3d_obj_load (argv[1], mesh);
  } else
    mesh3d_obj_build_default (mesh);

  mesh3d_obj_partition (mesh, (nb_patches == -1 ? 2 : nb_patches), MESH3D_PART_USE_SCOTCH | MESH3D_PART_SHOW_FRONTIERS);
}

int main (int argc, char *argv[])
{
  mesh3d_obj_init (&mesh);
  load_mesh (argc, argv, &mesh);

  mesh3d_obj_store ("output.obj", &mesh, 1);

  mesh3d_init (NULL);

  // Create SDL windows and initialize OpenGL context
  ctx[0] = mesh3d_ctx_create ("Mesh", SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_UNDEFINED, SCR_WIDTH, SCR_HEIGHT,
                              MESH3D_ENABLE_PICKING | MESH3D_ENABLE_HUD |
                                  MESH3D_ENABLE_CLIPPING);
  hud    = mesh3d_hud_alloc (ctx[0]);
  mesh3d_hud_toggle (ctx[0], hud);

  // Attach mesh
  mesh3d_set_mesh (ctx[0], &mesh);

  mesh3d_configure_data_colors_predefined (ctx[0], MESH3D_PALETTE_RAINBOW);

  float values[mesh.nb_cells];

  for (int i = 0; i < mesh.nb_cells; i++)
    values[i] = (float)mesh3d_obj_get_patch_of_cell (&mesh, i) /
                (float)(mesh.nb_patches - 1);

  mesh3d_set_data_colors (ctx[0], values);

  // render loop
  while (1) {
    process_events ();
    mesh3d_render (ctx, nb_ctx);
  }

  return 0;
}
