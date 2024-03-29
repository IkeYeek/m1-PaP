
#include "easypap.h"

#include <omp.h>
#include <fcntl.h>
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
      cur_data (c) = 0.0;

  // Stop after first iteration
  return 1;
}

int sample3d_do_patch_default (int start_cell, int end_cell)
{
  for (int c = start_cell; c < end_cell; c++)
    // Assign a distinct value to each cell
    cur_data (c) = (float)(c) / (float)(mesh.nb_cells - 1);

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
  mesh3d_configure_data_colors (ctx[0], colors, 2);

  if (picking_enabled) {
    debug_hud = mesh3d_hud_alloc (ctx[0]);
    mesh3d_hud_on (ctx[0], debug_hud);
  }
}

void sample3d_debug (int cell)
{
  if (cell == -1)
    mesh3d_hud_set (ctx[0], debug_hud, NULL);
  else
    mesh3d_hud_set (ctx[0], debug_hud, "Value: %f", cur_data (cell));
}


///////////////////////////// OpenCL version (ocl)
// Suggested cmdline(s):
// ./run -lm <your mesh file> -k sample3d -g -si
//
unsigned sample3d_invoke_ocl (unsigned nb_iter)
{
  size_t global[1] = {GPU_SIZE}; // global domain size for our calculation
  size_t local[1]  = {TILE};     // local domain size for our calculation
  cl_int err;

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 1, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, NB_CELLS, 0,
                       easypap_gpu_lane (TASK_TYPE_COMPUTE));

  // Stop after first iteration
  return 1;
}
