
#include "easypap.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void life3d_init (void)
{
  PRINT_DEBUG ('u', "Mesh size: %d\n", NB_CELLS);
  PRINT_DEBUG ('u', "#Patches: %d\n", NB_PATCHES);
  PRINT_DEBUG ('u', "Min cell neighbors: %d\n", min_neighbors ());
  PRINT_DEBUG ('u', "Max cell neighbors: %d\n", max_neighbors ());
}

// The Mesh is a one-dimension array of cells of size NB_CELLS. Each cell value
// is of type 'float' and should be kept between 0.0 and 1.0.

///////////////////////////// Sequential version (seq)
// Suggested cmdline(s):
// ./run -lm <your mesh file> -k life3d -si
//
unsigned life3d_compute_seq (unsigned nb_iter)
{
  int change = 0;
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (int p = 0; p < NB_PATCHES; p++) {
      do_patch (p);
    }
    swap_data ();
  }
  return change;
}

int life3d_do_patch_default (int start_cell, int end_cell)
{
  int change = 0;
  for (int c = 0; c < NB_CELLS; c++) {
    float me = cur_data (c);
    int nb_n = 0;
    for (int n = neighbor_start (c); n < neighbor_end (c); n++) {
      if (cur_data (neighbor (n)))
        nb_n++;
    }
    if (me == 1 && nb_n != 2 && nb_n != 3) {
      me     = 0.f;
      change = 1;
    } else if (me == 0 && nb_n == 3) {
      me     = 1.f;
      change = 1;
    }
    next_data (c) = me;
  }
  return change;
}

int life3d_do_patch_test (int start_cell, int end_cell)
{
  int change = 1;

  for (int c = 0; c < NB_CELLS; c++) {
    int cur = c;

    int me = cur_data (c);

    int top       = neighbor (neighbor_start (cur));
    int top_right = neighbor (neighbor_start (top) + 1);
    int top_left  = neighbor (neighbor_start (top) + 3);

    int right = neighbor (neighbor_start (cur) + 1);
    int left  = neighbor (neighbor_start (cur) + 3);

    int bottom       = neighbor (neighbor_start (cur) + 2);
    int bottom_right = neighbor (neighbor_start (bottom) + 1);
    int bottom_left  = neighbor (neighbor_start (bottom) + 3);

    int n = cur_data (top) + cur_data (top_right) + cur_data (top_left) +
            cur_data (right) + cur_data (left) + cur_data (bottom) +
            cur_data (bottom_right) + cur_data (bottom_left);

    const char new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    next_data (c)     = new_me;
    change |= me != new_me;
  }
  return change;
}

///////////////////////////// "tiled" version (tiled)
// Suggested cmdline(s):
// ./run -lm <your mesh file> -k life3d -si
//
unsigned life3d_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (int p = 0; p < NB_PATCHES; p++)
      do_patch (p);
    swap_data ();
  }

  // Stop after first iteration
  return 1;
}

#ifdef ENABLE_OPENCL

///////////////////////////// OpenCL version (ocl)
// Suggested cmdline(s):
// ./run -lm <your mesh file> -k life3d -g
//
unsigned life3d_compute_ocl (unsigned nb_iter)
{
  size_t global[1] = {GPU_SIZE}; // global domain size for our calculation
  size_t local[1]  = {TILE};     // local domain size for our calculation
  cl_int err;

  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                           &ocl_cur_buffer (0));
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 1,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");
  }

  clFinish (ocl_queue (0));

  monitoring_end_tile (0, 0, NB_CELLS, 0, easypap_gpu_lane (0));

  // Stop after first iteration
  return 1;
}

#endif // ENABLE_OPENCL

///////////////////////////// Initial config
static int debug_hud = -1;

void life3d_config (char *param)
{
  if (easypap_mesh_file == NULL)
    exit_with_error ("kernel %s needs a mesh (use --load-mesh <filename>)",
                     kernel_name);

  // Choose color palette
  float colors[] = {0.f, 0.f, 0.f, 1.f,  // dead
                    1.f, 1.f, 0.f, 1.f}; // alive

  mesh_data_set_palette (colors, 2);

  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], debug_hud);
  }
}
void life3d_draw_random (void)
{
  int nb_spots, spot_size;

  if (NB_CELLS >= 100)
    nb_spots = NB_CELLS / 100;
  else
    nb_spots = 1;
  spot_size = NB_CELLS / nb_spots / 4;
  if (!spot_size)
    spot_size = 1;

  for (int s = 0; s < nb_spots; s++) {
    int cell = random () % (NB_CELLS - spot_size);

    for (int c = cell; c < cell + spot_size; c++)
      cur_data (c) = 1.0;
  }
}
void life3d_draw (char *param)
{
  hooks_draw_helper (param, life3d_draw_random);
}
void life3d_debug (int cell)
{
  if (cell == -1)
    ezv_hud_set (ctx[0], debug_hud, NULL);
  else
    ezv_hud_set (ctx[0], debug_hud, "Value: %f", cur_data (cell));
}

static void set_partition_color (unsigned part, uint32_t color)
{
  ezv_set_cpu_color_1D (ctx[0], patch_start (part), patch_size (part), color);
}

static void set_partition_neighbors_color (unsigned part, uint32_t color)
{
  for (int ni = patch_neighbor_start (part); ni < patch_neighbor_end (part);
       ni++)
    set_partition_color (patch_neighbor (ni), color);
}

void life3d_overlay (int cell)
{
  // Example which shows how to highlight both selected cell and selected
  // partition
  int part = mesh3d_obj_get_patch_of_cell (&easypap_mesh_desc, cell);

  // highlight partition
  set_partition_color (part, ezv_rgb (255, 255, 255));
  // highlight neighbors of partition
  set_partition_neighbors_color (part, ezv_rgb (128, 128, 128));
  // highlight cell
  ezv_set_cpu_color_1D (ctx[0], cell, 1, ezv_rgb (50, 50, 50));
}
