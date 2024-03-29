#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cppdefs.h"
#include "debug.h"
#include "error.h"
#include "global.h"
#include "hooks.h"
#include "mesh3d.h"
#include "mesh3d_renderer.h"
#include "mesh_data.h"
#include "minmax.h"
#include "trace_record.h"
#include "gpu.h"

float *RESTRICT mesh_data     = NULL;
float *RESTRICT alt_mesh_data = NULL;

int *RESTRICT neighbors_soa = NULL;
int neighbor_soa_offset     = 0;

unsigned NB_CELLS   = 0;
unsigned NB_PATCHES = 0;

mesh3d_obj_t mesh;

#define MAX_CTX 2

mesh3d_ctx_t ctx[MAX_CTX] = {NULL, NULL};
unsigned nb_ctx           = 0;

// shared with main.c
unsigned picking_enabled = 0;

static int picked_cell   = -1;
static int iteration_hud = -1;
static int cell_hud      = -1;
static int partition_hud = -1;

void mesh_data_init (void)
{
  mesh3d_obj_init (&mesh);
  mesh3d_obj_load (easypap_mesh_file, &mesh);

  NB_CELLS = mesh.nb_cells;
}

void mesh_data_set_palette (float *data, unsigned size)
{
  if (do_display) {
    if (nb_ctx < 1)
      exit_with_error ("No mesh3d ctx created yet");

    mesh3d_configure_data_colors (ctx[0], data, size);
  }
}

void mesh_data_set_palette_predefined (mesh3d_palette_name_t palette)
{
  if (do_display) {
    if (nb_ctx < 1)
      exit_with_error ("No mesh3d ctx created yet");

    mesh3d_configure_data_colors_predefined (ctx[0], palette);
  }
#ifdef ENABLE_TRACE
  if (trace_may_be_used && palette != MESH3D_PALETTE_CUSTOM)
    trace_record_set_palette (palette);
#endif
}

void mesh_data_init_hud (int show)
{
  iteration_hud = mesh3d_hud_alloc (ctx[0]);
  if (show)
    mesh3d_hud_on (ctx[0], iteration_hud);

  if (picking_enabled) {
    cell_hud      = mesh3d_hud_alloc (ctx[0]);
    partition_hud = mesh3d_hud_alloc (ctx[0]);
  }
}

void mesh_data_toggle_hud (void)
{
  mesh3d_hud_toggle (ctx[0], iteration_hud);
}

void mesh_data_refresh (unsigned iter)
{
  mesh3d_hud_set (ctx[0], iteration_hud, "It: %d", iter);

  if (picking_enabled && the_picking != NULL)
    the_picking (picked_cell);

  // If computations were performed on CPU (that is, in mesh_data), copy data
  // into texture buffer Otherwise (GPU), data are already in place
  if (!gpu_used || !easypap_gl_buffer_sharing)
    mesh3d_set_data_colors (ctx[0], mesh_data);

  mesh3d_render (ctx, nb_ctx);
}

void mesh_data_process_event (SDL_Event *event, int *refresh)
{
  int pick;
  mesh3d_process_event (ctx, nb_ctx, event, refresh, &pick);
  if (picking_enabled && pick)
    mesh_data_do_pick ();
}

void mesh_data_alloc (void)
{
  const unsigned size = NB_CELLS * sizeof (float);

  mesh_data = mmap (NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mesh_data == NULL)
    exit_with_error ("Cannot allocate mesh data: mmap failed");

  alt_mesh_data = mmap (NULL, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (alt_mesh_data == NULL)
    exit_with_error ("Cannot allocate alternate mesh data: mmap failed");

  PRINT_DEBUG ('i', "Init phase 4: mesh data allocated\n");
}

void mesh_data_free (void)
{
  const unsigned size = NB_CELLS * sizeof (float);

  if (mesh_data != NULL) {
    munmap (mesh_data, size);
    mesh_data = NULL;
  }

  if (alt_mesh_data != NULL) {
    munmap (alt_mesh_data, size);
    alt_mesh_data = NULL;
  }
}

void mesh_data_replicate (void)
{
  memcpy (alt_mesh_data, mesh_data, NB_CELLS * sizeof (float));
}

void mesh_data_dump_to_file (char *filename)
{
  int fd = open (filename, O_CREAT | O_TRUNC | O_WRONLY, 0666);
  if (fd == -1)
    exit_with_error ("Cannot open %s (%s)", filename, strerror (errno));

  int n = write (fd, mesh_data, NB_CELLS * sizeof (float));
  if (n < NB_CELLS * sizeof (float))
    exit_with_error ("Could not write thumbnail data");

  close (fd);
}

void mesh_data_save_thumbnail (unsigned iteration)
{
  char filename[1024];

  sprintf (filename, "%s/thumb_%04d.raw", DEFAULT_EZV_TRACE_DIR, iteration);

  mesh_data_dump_to_file (filename);
}

void mesh_data_build_neighbors_soa (unsigned round)
{
  if (neighbors_soa != NULL)
    return; // structure already built

  if (mesh.max_neighbors == 0)
    exit_with_error ("Mesh not yet initialized");

  neighbor_soa_offset = ROUND_TO_MULTIPLE (mesh.nb_cells, round);
  const unsigned size = neighbor_soa_offset * mesh.max_neighbors * sizeof (int);

  neighbors_soa = malloc (size);
  int index     = 0;
  for (int c = 0; c < mesh.nb_cells; c++) {
    int n = 0;
    while (index < mesh.index_first_neighbor[c + 1]) {
      neighbors_soa[n * neighbor_soa_offset + c] = mesh.neighbors[index];
      index++;
      n++;
    }
    while (n < mesh.max_neighbors) {
      neighbors_soa[n * neighbor_soa_offset + c] = -1;
      n++;
    }
  }
  PRINT_DEBUG ('m', "Neighbors SOA built (rounded to multiple of %d)\n", round);
}

void mesh_data_do_pick (void)
{
  int p = mesh3d_perform_picking (ctx, 1);

  if (p != picked_cell) { // focus has changed
    picked_cell = p;

    mesh3d_reset_cpu_colors (ctx[0]);

    if (p != -1) {
      mesh3d_hud_on (ctx[0], cell_hud);
      mesh3d_hud_set (ctx[0], cell_hud, "Cell: %d", p);

      int partoche = mesh3d_obj_get_patch_of_cell (&mesh, p);

      mesh3d_hud_on (ctx[0], partition_hud);
      mesh3d_hud_set (ctx[0], partition_hud, "Part: %d", partoche);

      // subdomain:
      mesh3d_set_cpu_color (ctx[0], mesh.patch_first_cell[partoche],
                            mesh.patch_first_cell[partoche + 1] -
                                mesh.patch_first_cell[partoche],
                            0xFFFFFFC0);
      // cell:
      mesh3d_set_cpu_color (ctx[0], p, 1, 0xFF0000C0);
    } else {
      mesh3d_hud_off (ctx[0], cell_hud);
      mesh3d_hud_off (ctx[0], partition_hud);
    }
  }
}