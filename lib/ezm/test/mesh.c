
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "ezm.h"
#include "ezv.h"
#include "ezv_event.h"

// settings
const unsigned int WIN_WIDTH  = 1024;
const unsigned int WIN_HEIGHT = 768;

#define MAX_CTX 3

static float *mesh_values = NULL;

static char *mesh_file = "../../data/mesh/1-torus.obj";

static mesh3d_obj_t mesh;
static ezv_ctx_t ctx[MAX_CTX] = {NULL, NULL, NULL};
static unsigned nb_ctx        = 0;

// profiling
static ezm_recorder_t recorder = NULL;

static unsigned do_trace = 0;

// Debug HUDs
static int pos_hud = -1;
static int val_hud = -1;

static void create_huds (ezv_ctx_t ctx)
{
  pos_hud = ezv_hud_alloc (ctx);
  ezv_hud_on (ctx, pos_hud);
  val_hud = ezv_hud_alloc (ctx);
  ezv_hud_on (ctx, val_hud);
}

static void do_pick (void)
{
  int p = ezv_perform_1D_picking (ctx, nb_ctx);

  ezv_reset_cpu_colors (ctx[0]);

  if (p == -1) {
    ezv_hud_off (ctx[0], pos_hud);
    ezv_hud_off (ctx[0], val_hud);
  } else {
    float v = mesh_values[p];

    ezv_hud_on (ctx[0], pos_hud);
    ezv_hud_set (ctx[0], pos_hud, "Cell: %d", p);
    ezv_hud_on (ctx[0], val_hud);
    ezv_hud_set (ctx[0], val_hud, "Value: %f", v);

    int partoche = mesh3d_obj_get_patch_of_cell (&mesh, p);
    // Set partition color
    ezv_set_cpu_color_1D (ctx[0], mesh.patch_first_cell[partoche],
                          mesh.patch_first_cell[partoche + 1] -
                              mesh.patch_first_cell[partoche],
                          ezv_rgba (0xFF, 0xFF, 0xFF, 0xE0));
    // Set cell color
    ezv_set_cpu_color_1D (ctx[0], p, 1, ezv_rgba (0xFF, 0x00, 0x00, 0xFF));
  }
}

static void process_events (int blocking)
{
  SDL_Event event;
  int r;

  do {
    r = ezv_get_event (&event, blocking);
    if (r > 0) {
      switch (event.type) {
      case SDL_KEYDOWN:
        switch (event.key.keysym.sym) {
        case SDLK_h:
          ezm_recorder_toggle_heat_mode (recorder);
          break;
        case SDLK_e:
          ezm_recorder_enable (recorder, 0);
          break;
        case SDLK_d:
          ezm_recorder_disable (recorder);
          break;
        default:
          ezv_process_event (ctx, nb_ctx, &event, NULL, NULL);
        }
        break;
      default:
        ezv_process_event (ctx, nb_ctx, &event, NULL, NULL);
      }
    }
  } while (r > 0 && !blocking);
  do_pick ();
}

static void config_init ();
static unsigned nb_openmp_threads (void);

#define MAX(a, b)                                                              \
  ({                                                                           \
    __typeof__ (a) _a = (a);                                                   \
    __typeof__ (b) _b = (b);                                                   \
    _a > _b ? _a : _b;                                                         \
  })

int do_one_patch (int start_cell, int end_cell)
{
  int change = 0;

  for (int c = start_cell; c < end_cell; c++) {
    float v = mesh_values[c];

    for (int n = mesh.index_first_neighbor[c];
         n < mesh.index_first_neighbor[c + 1]; n++)
      v = MAX (v, mesh_values[mesh.neighbors[n]]);
    if (v > mesh_values[c]) {
      mesh_values[c] = v;
      change         = 1;
    }
  }

  return change;
}

int do_patch (int p)
{
  int me = omp_get_thread_num ();

  ezm_start_work (recorder, me);

  int change =
      do_one_patch (mesh.patch_first_cell[p], mesh.patch_first_cell[p + 1]);

  ezm_end_1D (recorder, me, mesh.patch_first_cell[p],
              mesh.patch_first_cell[p + 1] - mesh.patch_first_cell[p]);

  return change;
}

int compute (void)
{
  unsigned change = 0;

  ezm_start_iteration (recorder);

#pragma omp parallel for schedule(runtime) reduction(| : change)
  for (int p = 0; p < mesh.nb_patches; p++)
    change |= do_patch (p);

  ezm_end_iteration (recorder);

  return change;
}

int main (int argc, char *argv[])
{
  if (argc > 1 && strcmp (argv[1], "-t") == 0) {
    do_trace = 1;
    argc--;
    argv++;
  }

  ezm_init (NULL, do_trace ? EZM_NO_DISPLAY : 0);

  // Initialize mesh
  mesh3d_obj_init (&mesh);
  if (argc > 1)
    mesh_file = argv[1];

  mesh3d_obj_load (mesh_file, &mesh);

  // data values (e.g. temperature)
  mesh_values = calloc (mesh.nb_cells, sizeof (float));

  if (!do_trace) {
    // Create SDL windows and initialize OpenGL context
    ctx[nb_ctx++] = ezv_ctx_create (
        EZV_CTX_TYPE_MESH3D, "Mesh", 0, 0, WIN_WIDTH, WIN_HEIGHT,
        EZV_ENABLE_PICKING | EZV_ENABLE_HUD | EZV_ENABLE_CLIPPING);
    // Attach mesh
    ezv_mesh3d_set_mesh (ctx[0], &mesh);

    ezv_use_cpu_colors (ctx[0]);
    ezv_use_data_colors_predefined (ctx[0], EZV_PALETTE_RAINBOW);

    // Create head-up displays
    create_huds (ctx[0]);
  }

  // Profiling
  recorder = ezm_recorder_create (nb_openmp_threads (), 0);
  if (do_trace) {
    ezm_recorder_attach_tracerec (recorder, "max.evt", "CC Labelling (OpenMP)");
    ezm_recorder_store_mesh3d_filename (recorder, mesh_file);
  } else {
    ezm_set_cpu_palette (recorder, EZV_PALETTE_RAINBOW, 0);
    ezm_helper_add_perfmeter (recorder, ctx, &nb_ctx);
    ezm_helper_add_footprint (recorder, ctx, &nb_ctx);
  }
  ezm_recorder_enable (recorder, 1);

  config_init ();

  int change = 1;
  int it     = 0;
  while (1) {
    // compute one image
    if (change) {
      change = compute ();
      it++;
      if (!do_trace)
        ezv_set_data_colors (ctx[0], mesh_values);
    }

    if (!do_trace) {
      // check for UI events
      process_events (!change);

      // refresh display
      ezv_render (ctx, nb_ctx);
    } else if (!change)
      break;
  }

  printf ("Completed after %d iterations\n", it);

  ezm_recorder_destroy (recorder);

  free (mesh_values);

  return 0;
}

////////////////////////
static unsigned nb_openmp_threads (void)
{
  unsigned n;
#pragma omp parallel
  {
#pragma omp single
    n = omp_get_num_threads ();
  }
  return n;
}

// Heat data
static void config_init ()
{
  for (int c = 0; c < mesh.nb_cells; c++)
    mesh_values[c] = (float)c / (float)(mesh.nb_cells - 1);
}
