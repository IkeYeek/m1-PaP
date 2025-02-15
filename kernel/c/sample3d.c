
#include "easypap.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void sample3d_init (void)
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
// ./run -lm <your mesh file> -k sample3d -si
//
unsigned sample3d_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    for (int c = 0; c < NB_CELLS; c++)
      cur_data (c) = 0.5;

  // Stop after first iteration
  return 1;
}

int sample3d_do_patch_default (int start_cell, int end_cell)
{
  for (int c = start_cell; c < end_cell; c++)
    // Assign a distinct value to each cell
    cur_data (c) = (float)(c) / (float)(NB_CELLS - 1);

  return 0;
}

///////////////////////////// "tiled" version (tiled)
// Suggested cmdline(s):
// ./run -lm <your mesh file> -k sample3d -si
//
unsigned sample3d_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    for (int p = 0; p < NB_PATCHES; p++)
      do_patch (p);

  // Stop after first iteration
  return 1;
}

#ifdef ENABLE_OPENCL

///////////////////////////// OpenCL version (ocl)
// Suggested cmdline(s):
// ./run -lm <your mesh file> -k sample3d -g
//
unsigned sample3d_compute_ocl (unsigned nb_iter)
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

void sample3d_config (char *param)
{
  if (easypap_mesh_file == NULL)
    exit_with_error ("kernel %s needs a mesh (use --load-mesh <filename>)",
                     kernel_name);

  // Choose color palette
  float colors[] = {0.0f, 0.8f, 0.8f, 1.f,  // cyan
                    0.8f, 0.0f, 0.8f, 1.f}; // pink

  mesh_data_set_palette (colors, 2);

  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], debug_hud);
  }
}

void sample3d_debug (int cell)
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

void sample3d_overlay (int cell)
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
