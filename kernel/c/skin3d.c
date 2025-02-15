
#include "easypap.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static int debug_hud = -1;

void skin3d_config (char *param)
{
  if (easypap_mesh_file == NULL)
    exit_with_error ("kernel %s needs a mesh (use --load-mesh <filename>)",
                     kernel_name);
  int palette = EZV_PALETTE_RAINBOW;

  if (param != NULL) {
    int n = atoi (param);
    if (n >= EZV_PALETTE_LINEAR && n <= EZV_PALETTE_RAINBOW)
      palette = n;
  }
  // Choose color palette
  mesh_data_set_palette_predefined (palette);

  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], debug_hud);
  }
}

void skin3d_debug (int cell)
{
  if (cell == -1)
    ezv_hud_set (ctx[0], debug_hud, NULL);
  else
    ezv_hud_set (ctx[0], debug_hud, "Value: %f", cur_data (cell));
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
  for (int c = 0; c < NB_CELLS; c++)
    cur_data (c) = (float)(c) / (float)(NB_CELLS - 1);
}

void skin3d_draw_partitions (void)
{
  for (int c = 0; c < NB_CELLS; c++)
    cur_data (c) = (float)mesh3d_obj_get_patch_of_cell (&easypap_mesh_desc, c) /
                   (float)(NB_PATCHES - 1);
}

static void draw_coord (int coord)
{
  // Color cells gradually from 0 to 1 along <coord> axis
  for (int c = 0; c < NB_CELLS; c++) {
    bbox_t box;
    mesh3d_obj_get_bbox_of_cell (&easypap_mesh_desc, c, &box);
    float f = (box.min[coord] + box.max[coord]) / 2.0f;
    cur_data (c) =
        (f - easypap_mesh_desc.bbox.min[coord]) /
        (easypap_mesh_desc.bbox.max[coord] - easypap_mesh_desc.bbox.min[coord]);
  }
}

void skin3d_draw_x (void)
{
  draw_coord (0);
}

void skin3d_draw_y (void)
{
  draw_coord (1);
}

void skin3d_draw_z (void)
{
  draw_coord (2);
}

static void draw_uniform (float value)
{
  for (int c = 0; c < NB_CELLS; c++)
    cur_data (c) = value;
}

void skin3d_draw_uni (void)
{
  draw_uniform (0.65f);
}

void skin3d_draw (char *param)
{
  if (param) {
    if (strstr (param, ".raw") != NULL && (access (param, R_OK) != -1)) {
      // The parameter is a filename, so we guess it's a raw dump file
      int fd = open (param, O_RDONLY);
      if (fd == -1)
        exit_with_error ("Cannot open %s (%s)", param, strerror (errno));

      int n = read (fd, mesh_data, NB_CELLS * sizeof (float));
      if (n < NB_CELLS * sizeof (float))
        exit_with_error ("Could not read enough data from raw file");

      close (fd);
      return;
    } else {
      // Check if the parameter is a float
      int len;
      float value;
      int ret = sscanf (param, "%f %n", &value, &len);
      if (ret == 1 && !param[len]) {
        // The parameter is a float, so we use it as a value for all cells
        draw_uniform (value);
        return;
      }
    }
  }
  // Call function ${kernel}_draw_${param}, or default function (second
  // parameter) if symbol not found
  hooks_draw_helper (param, skin3d_draw_cells);
}
