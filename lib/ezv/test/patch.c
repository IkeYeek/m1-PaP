
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

static float *data  = NULL;
static int datasize = 0;

static inline unsigned *cell_neighbors (int cell)
{
  unsigned i = mesh.index_first_neighbor[cell];
  return mesh.neighbors + i;
}

static void set_value_for_partition (unsigned part, float value)
{
  for (int c = mesh.patch_first_cell[part]; c < mesh.patch_first_cell[part + 1];
       c++)
    data[c] = value;
}

static void set_value_for_neighbors (unsigned part, float value)
{
  for (int ni = mesh.index_first_patch_neighbor[part];
       ni < mesh.index_first_patch_neighbor[part + 1]; ni++) {
    int p = mesh.patch_neighbors[ni];
    set_value_for_partition (p, value);
  }
}

static void do_pick (void)
{
  int p = ezv_perform_1D_picking (ctx, nb_ctx);

  memset (data, 0, datasize);

  if (p != -1) {
    int partoche = mesh3d_obj_get_patch_of_cell (&mesh, p);
    ezv_hud_set (ctx[0], hud, "Part: %d", partoche);

    set_value_for_partition (partoche, 0.25);

    if (mesh.patch_neighbors != NULL)
      set_value_for_neighbors (partoche, 0.75);

    data[p] = 1.0; // selected cell
  } else
    ezv_hud_set (ctx[0], hud, NULL);

  ezv_set_data_colors (ctx[0], data);
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

static void load_mesh (int argc, char *argv[], mesh3d_obj_t *mesh)
{
  int nb_patches = -1;

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
    else if (!strcmp (argv[1], "--wall") || !strcmp (argv[1], "-w"))
      mesh3d_obj_build_wall (mesh, 8);
    else
      mesh3d_obj_load (argv[1], mesh);
  } else
    mesh3d_obj_build_default (mesh);

  if (nb_patches == -1) {
    if (mesh->nb_patches == 0)
      mesh3d_obj_partition (
          mesh, 2, MESH3D_PART_USE_SCOTCH | MESH3D_PART_SHOW_FRONTIERS);
  } else {
    mesh3d_obj_partition (mesh, nb_patches,
                          MESH3D_PART_USE_SCOTCH | MESH3D_PART_SHOW_FRONTIERS);
  }
}

int main (int argc, char *argv[])
{
  ezv_init (NULL);

  mesh3d_obj_init (&mesh);
  load_mesh (argc, argv, &mesh);

  // Create SDL windows and initialize OpenGL context
  ctx[0] = ezv_ctx_create (EZV_CTX_TYPE_MESH3D, "Patch", SDL_WINDOWPOS_CENTERED,
                           SDL_WINDOWPOS_UNDEFINED, SCR_WIDTH, SCR_HEIGHT,
                           EZV_ENABLE_PICKING | EZV_ENABLE_HUD |
                               EZV_ENABLE_CLIPPING);
  hud    = ezv_hud_alloc (ctx[0]);
  ezv_hud_toggle (ctx[0], hud);

  // Attach mesh
  ezv_mesh3d_set_mesh (ctx[0], &mesh);

  // Heat Palette (as simple as defining these five key colors)
  float colors[] = {0.0f, 0.0f, 0.0f, 0.0f,  // transparent
                    0.0f, 1.0f, 1.0f, 1.0f,  // cyan
                    0.0f, 1.0f, 0.0f, 1.0f,  // green
                    1.0f, 1.0f, 0.0f, 1.0f,  // yellow
                    1.0f, 0.0f, 0.0f, 1.0f}; // red

  ezv_use_data_colors (ctx[0], colors, 5);

  datasize = mesh.nb_cells * sizeof (float);
  data     = malloc (datasize);

  memset (data, 0, datasize);

  ezv_set_data_colors (ctx[0], data);

  // render loop
  while (1) {
    process_events ();
    ezv_render (ctx, nb_ctx);
  }

  return 0;
}
