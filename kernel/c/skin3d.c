
#include "easypap.h"

#include <omp.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int debug_hud = -1;

void skin3d_config (char *param)
{
  if (easypap_mesh_file == NULL)
    exit_with_error ("kernel %s needs a mesh (use --load-mesh <filename>)",
                     kernel_name);

  // Choose color palette
  mesh_data_set_palette_predefined (MESH3D_PALETTE_RAINBOW);

  if (picking_enabled) {
    debug_hud = mesh3d_hud_alloc (ctx[0]);
    mesh3d_hud_on (ctx[0], debug_hud);
  }
}

void skin3d_debug (int cell)
{
  if (cell == -1)
    mesh3d_hud_set (ctx[0], debug_hud, NULL);
  else
    mesh3d_hud_set (ctx[0], debug_hud, "Value: %f", cur_data (cell));
}

void skin3d_init (void)
{
  PRINT_DEBUG ('u', "Mesh size: %d\n", NB_CELLS);
  PRINT_DEBUG ('u', "#Patches: %d\n", NB_PATCHES);
  PRINT_DEBUG ('u', "Min cell neighbors: %d\n", min_neighbors ());
  PRINT_DEBUG ('u', "Max cell neighbors: %d\n", max_neighbors ());
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run -lm <your mesh file> -k skin3d -a <your raw dump file>
//
unsigned skin3d_compute_seq (unsigned nb_iter)
{
  // Do nothing
  return 1;
}

///////////////////////////// Initial configuration

void skin3d_draw_cells (void)
{
  // Unique color per cell
  for (int c = 0; c < mesh.nb_cells; c++)
    mesh_data[c] = (float)(c) / (float)(mesh.nb_cells - 1);
}

void skin3d_draw_partitions (void)
{
  for (int c = 0; c < mesh.nb_cells; c++)
    mesh_data[c] = (float)mesh3d_obj_get_patch_of_cell (&mesh, c) /
                   (float)(mesh.nb_patches - 1);
}

void skin3d_draw_z (void)
{
  // Color cells gradually from 0 to 1 along z-axis
  const int COORD = 2; // z-axis
  for (int c = 0; c < mesh.nb_cells; c++) {
    bbox_t box;
    mesh3d_obj_get_bbox_of_cell (&mesh, c, &box);
    float z      = (box.min[COORD] + box.max[COORD]) / 2.0f;
    mesh_data[c] = (z - mesh.bbox.min[COORD]) /
                   (mesh.bbox.max[COORD] - mesh.bbox.min[COORD]);
  }
}

void skin3d_draw (char *param)
{
  if (param && strstr (param, ".raw") != NULL && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a raw dump file
    int fd = open (param, O_RDONLY);
    if (fd == -1)
      exit_with_error ("Cannot open %s (%s)", param, strerror (errno));

    int n = read (fd, mesh_data, NB_CELLS * sizeof (float));
    if (n < NB_CELLS * sizeof (float))
      exit_with_error ("Could not read enough data from raw file");

    close (fd);
    return;
  }

  // Call function ${kernel}_draw_${param}, or default function (second
  // parameter) if symbol not found
  hooks_draw_helper (param, skin3d_draw_cells);
}
