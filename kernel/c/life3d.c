
#include "easypap.h"

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static char *curr_dirty = 0;
static char *next_dirty = 0;

void life3d_init (void)
{
  const int size = sizeof (char) * NB_PATCHES;
  curr_dirty     = malloc (size);
  next_dirty     = malloc (size);

  for (int i = 0; i < size; i++) {
    curr_dirty[i] = 1;
    next_dirty[i] = 0;
  }

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

unsigned life3d_compute_ompfor (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
#pragma omp parallel for
    for (int p = 0; p < NB_PATCHES; p++)
      do_patch (p);
    swap_data ();
  }

  // Stop after first iteration
  return 0;
}

unsigned life3d_compute_lazy (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
#pragma omp parallel for
    for (int p = 0; p < NB_PATCHES; p++) {
      if (!curr_dirty[p])
        continue;
      int local_change = do_patch (p);
    }
    swap_data ();
  }

  // Stop after first iteration
  return 0;
}

#ifdef ENABLE_OPENCL

static cl_mem neighbors_buffer = 0, index_buffer = 0;

void life3d_init_ocl_naive (void)
{
  cl_int err;

  // Array of all neighbors
  const int sizen = easypap_mesh_desc.total_neighbors * sizeof (unsigned);

  neighbors_buffer =
      clCreateBuffer (context, CL_MEM_READ_WRITE, sizen, NULL, NULL);
  if (!neighbors_buffer)
    exit_with_error ("Failed to allocate neighbor buffer");

  err =
      clEnqueueWriteBuffer (ocl_queue (0), neighbors_buffer, CL_TRUE, 0, sizen,
                            easypap_mesh_desc.neighbors, 0, NULL, NULL);
  check (err, "Failed to write to neighbor buffer");

  // indexes
  const int sizei = (NB_CELLS + 1) * sizeof (unsigned);

  index_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizei, NULL, NULL);
  if (!index_buffer)
    exit_with_error ("Failed to allocate index buffer");

  err = clEnqueueWriteBuffer (ocl_queue (0), index_buffer, CL_TRUE, 0, sizei,
                              easypap_mesh_desc.index_first_neighbor, 0, NULL,
                              NULL);
  check (err, "Failed to write to index buffer");
}

unsigned life3d_compute_ocl_naive (unsigned nb_iter)
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
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                           &neighbors_buffer);
    err |= clSetKernelArg (ocl_compute_kernel (0), 3, sizeof (cl_mem),
                           &index_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 1,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp          = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
    }
  }

  clFinish (ocl_queue (0));

  monitoring_end_tile (0, 0, NB_CELLS, 0, easypap_gpu_lane (0));

  return 0;
}

#endif
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
    nb_spots = (NB_CELLS / 100) * 2;
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
