#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cppdefs.h"
#include "debug.h"
#include "error.h"
#include "ezp_ctx.h"
#include "ezv.h"
#include "global.h"
#include "gpu.h"
#include "hooks.h"
#include "mesh_data.h"
#include "img_data.h"
#include "minmax.h"

float *RESTRICT mesh_data     = NULL;
float *RESTRICT alt_mesh_data = NULL;

int *RESTRICT neighbors_soa = NULL;
int neighbor_soa_offset     = 0;

unsigned NB_CELLS   = 0;
unsigned NB_PATCHES = 0;

mesh3d_obj_t easypap_mesh_desc;

static int picked_cell   = -1;
static int cell_hud      = -1;
static int partition_hud = -1;
static int val_hud       = -1;

static ezv_palette_name_t the_data_palette = EZV_PALETTE_UNDEFINED;

static void check_patch_size (void)
{
  NB_PATCHES = NB_TILES_X;

  if (NB_PATCHES == 0) {
    if (TILE_W > 0) // -ts or -tw used
      NB_PATCHES = NB_CELLS / TILE_W;
    else {
      if (easypap_mesh_desc.nb_patches > 0)
        NB_PATCHES = 0;
      else
        NB_PATCHES = 2; // default = 2 patches
      use_scotch = 1;
    }
  }

  if (NB_PATCHES > NB_CELLS)
    exit_with_error ("NB_PATCHES (%d) is greater than NB_CELLS (%d)!",
                     NB_PATCHES, NB_CELLS);
}

void mesh_data_init (void)
{
  mesh3d_obj_init (&easypap_mesh_desc);
  mesh3d_obj_load (easypap_mesh_file, &easypap_mesh_desc);

  NB_CELLS = easypap_mesh_desc.nb_cells;

  check_patch_size ();

  PRINT_DEBUG ('i', "Init phase 0 (MESH3D mode) : NB_CELLS = %d\n", NB_CELLS);
}

void mesh_data_set_palette (float *data, unsigned size)
{
  if (do_display) {
    if (nb_ctx < 1)
      exit_with_error ("No mesh3d ctx created yet");

    ezv_use_data_colors (ctx[0], data, size);
  }

  the_data_palette = EZV_PALETTE_CUSTOM;
}

void mesh_data_set_palette_predefined (ezv_palette_name_t palette)
{
  if (do_display) {
    if (nb_ctx < 1)
      exit_with_error ("No mesh3d ctx created yet");

    ezv_use_data_colors_predefined (ctx[0], palette);
  }

  the_data_palette = palette;
}

void mesh_data_set_default_palette_if_none_defined (void)
{
  if (do_display) {
    if (nb_ctx < 1)
      exit_with_error ("No mesh3d ctx created yet");

    if (the_data_palette == EZV_PALETTE_UNDEFINED) {
      the_data_palette = EZV_PALETTE_RAINBOW;
      ezv_use_data_colors_predefined (ctx[0], the_data_palette);
    }
  }
}

ezv_palette_name_t mesh_data_get_palette (void)
{
  return the_data_palette;
}

void mesh_data_init_huds (int show)
{
  ezp_ctx_ithud_init (show);

  if (picking_enabled) {
    cell_hud      = ezv_hud_alloc (ctx[0]);
    partition_hud = ezv_hud_alloc (ctx[0]);
    if (the_1d_debug == NULL) {
      val_hud = ezv_hud_alloc (ctx[0]);
      ezv_hud_on (ctx[0], val_hud);
    }
  }
}

void mesh_data_refresh (unsigned iter)
{
  ezp_ctx_ithud_set (iter);

  if (picking_enabled) {
    if (the_1d_debug != NULL)
      the_1d_debug (picked_cell);
    else {
      if (picked_cell == -1)
        ezv_hud_set (ctx[0], val_hud, NULL);
      else {
        ezv_hud_set (ctx[0], val_hud, "Value: %f", mesh_data[picked_cell]);
      }
    }
  }

  // If computations were performed on CPU (that is, in mesh_data), copy data
  // into texture buffer Otherwise (GPU), data are already in place
  if (!gpu_used || !easypap_gl_buffer_sharing)
    ezv_set_data_colors (ctx[0], mesh_data);

  ezv_render (ctx, nb_ctx);
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

  if (easypap_mesh_desc.max_neighbors == 0)
    exit_with_error ("Mesh not yet initialized");

  neighbor_soa_offset = ROUND_TO_MULTIPLE (easypap_mesh_desc.nb_cells, round);
  const unsigned size =
      neighbor_soa_offset * easypap_mesh_desc.max_neighbors * sizeof (int);

  neighbors_soa = malloc (size);
  int index     = 0;
  for (int c = 0; c < easypap_mesh_desc.nb_cells; c++) {
    int n = 0;
    while (index < easypap_mesh_desc.index_first_neighbor[c + 1]) {
      neighbors_soa[n * neighbor_soa_offset + c] =
          easypap_mesh_desc.neighbors[index];
      index++;
      n++;
    }
    while (n < easypap_mesh_desc.max_neighbors) {
      neighbors_soa[n * neighbor_soa_offset + c] = -1;
      n++;
    }
  }
  PRINT_DEBUG ('m', "Neighbors SOA built (rounded to multiple of %d)\n", round);
}

static void mesh_data_do_pick (void)
{
  int p = ezv_perform_1D_picking (ctx, 1);

  if (p != picked_cell) { // focus has changed
    picked_cell = p;

    ezv_reset_cpu_colors (ctx[0]);

    if (p != -1) {
      ezv_hud_on (ctx[0], cell_hud);
      ezv_hud_set (ctx[0], cell_hud, "Cell: %d", p);

      int partoche = mesh3d_obj_get_patch_of_cell (&easypap_mesh_desc, p);

      ezv_hud_on (ctx[0], partition_hud);
      ezv_hud_set (ctx[0], partition_hud, "Patch: %d", partoche);

      if (the_1d_overlay != NULL)
        the_1d_overlay (p);
      else {
        // partition
        ezv_set_cpu_color_1D (ctx[0],
                              easypap_mesh_desc.patch_first_cell[partoche],
                              easypap_mesh_desc.patch_first_cell[partoche + 1] -
                                  easypap_mesh_desc.patch_first_cell[partoche],
                              ezv_rgba (0xFF, 0xFF, 0xFF, 0xC0));
        // cell
        ezv_set_cpu_color_1D (ctx[0], p, 1, ezv_rgba (0xFF, 0x00, 0x00, 0xC0));
      }
    } else {
      ezv_hud_off (ctx[0], cell_hud);
      ezv_hud_off (ctx[0], partition_hud);
    }
  }
}

void mesh_data_process_event (SDL_Event *event, int *refresh)
{
  int pick;
  ezv_process_event (ctx, nb_ctx, event, refresh, &pick);
  if (picking_enabled && pick)
    mesh_data_do_pick ();
}

void mesh_data_reorder_partitions (int newpos[])
{
  if (neighbors_soa != NULL)
    fprintf (stderr, "WARNING: Reordering of partitions/cells will not update "
                     "neighbor_soa data structures!");

  mesh3d_reorder_partitions (&easypap_mesh_desc, newpos);
  ezv_mesh3d_refresh_mesh (ctx, nb_ctx);

  ezv_render (ctx, nb_ctx);
}

void mesh_data_shuffle_cells (void)
{
  if (neighbors_soa != NULL)
    fprintf (stderr, "WARNING: Reordering of partitions/cells will not update "
                     "neighbor_soa data structures!");

  mesh3d_shuffle_cells_in_partitions (&easypap_mesh_desc);
  ezv_mesh3d_refresh_mesh (ctx, nb_ctx);

  ezv_render (ctx, nb_ctx);
}
