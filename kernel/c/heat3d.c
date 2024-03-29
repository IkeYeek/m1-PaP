
#include "easypap.h"

#include <omp.h>

static int debug_hud = -1;

void heat3d_config (char *param)
{
  if (easypap_mesh_file == NULL)
    exit_with_error ("kernel %s needs a mesh (use --load-mesh <filename>)",
                     kernel_name);

  // Choose color palette
  mesh_data_set_palette_predefined (MESH3D_PALETTE_HEAT);

  if (picking_enabled) {
    debug_hud = mesh3d_hud_alloc (ctx[0]);
    mesh3d_hud_on (ctx[0], debug_hud);
  }
}

void heat3d_debug (int cell)
{
  if (cell == -1)
    mesh3d_hud_set (ctx[0], debug_hud, "No selection");
  else
    mesh3d_hud_set (ctx[0], debug_hud, "Temp: %f", cur_data(cell));
}

void heat3d_init (void)
{
  PRINT_DEBUG ('u', "Mesh size: %d\n", NB_CELLS);
  PRINT_DEBUG ('u', "#Patches: %d\n", NB_PATCHES);
  PRINT_DEBUG ('u', "Min cell neighbors: %d\n", min_neighbors ());
  PRINT_DEBUG ('u', "Max cell neighbors: %d\n", max_neighbors ());
}

// The Mesh is a one-dimension array of cells of size NB_CELLS. Each cell value
// is of type 'float' and should be kept between 0.0 and 1.0.

int heat3d_do_patch_default (int start_cell, int end_cell)
{
  for (int c = start_cell; c < end_cell; c++) {
    for (int n = neighbor_start (c); n < neighbor_end (c); n++)
      // TODO
      ;
  }

  return 0;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run -lm 2-torus.cgns -k heat3d -v seq -a 10
//
unsigned heat3d_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int p = 0; p < NB_PATCHES; p++)
      do_patch (p);

    swap_data ();
  }

  return 0;
}


///////////////////////////// Initial configuration

void heat3d_draw_random (void)
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

void heat3d_draw_fifty (void)
{
  for (int c = 0; c < NB_CELLS >> 1; c++)
    cur_data (c) = 1.0;
}

static void draw_coord (int coord)
{
  const float mid = (mesh.bbox.min[coord] + mesh.bbox.max[coord]) / 2.0f;

  for (int c = 0; c < mesh.nb_cells; c++) {
    bbox_t box;
    mesh3d_obj_get_bbox_of_cell (&mesh, c, &box);
    float f   = (box.min[coord] + box.max[coord]) / 2.0f;
    cur_data (c) = (f <= mid) ? 0.0f : 1.0f;
  }
}

void heat3d_draw_x (void)
{
  draw_coord (0);
}

void heat3d_draw_y (void)
{
  draw_coord (1);
}

void heat3d_draw_z (void)
{
  draw_coord (2);
}

void heat3d_draw (char *param)
{
  // Call function ${kernel}_draw_${param}, or default function (second
  // parameter) if symbol not found
  hooks_draw_helper (param, heat3d_draw_random);
}

///////////////////////////// naive OpenCL

static cl_mem neighbors_buffer = 0, index_buffer = 0;

void heat3d_init_ocl_naive (void)
{
  cl_int err;

  // Array of all neighbors
  const int sizen = mesh.total_neighbors * sizeof (unsigned);

  neighbors_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizen, NULL, NULL);
  if (!neighbors_buffer)
    exit_with_error ("Failed to allocate neighbor buffer");

  err = clEnqueueWriteBuffer (queue, neighbors_buffer, CL_TRUE, 0, sizen, mesh.neighbors, 0,
                              NULL, NULL);
  check (err, "Failed to write to neighbor buffer");

  // indexes
  const int sizei = (mesh.nb_cells + 1) * sizeof (unsigned);

  index_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizei, NULL, NULL);
  if (!index_buffer)
    exit_with_error ("Failed to allocate index buffer");

  err = clEnqueueWriteBuffer (queue, index_buffer, CL_TRUE, 0, sizei, mesh.index_first_neighbor, 0,
                              NULL, NULL);
  check (err, "Failed to write to index buffer");
}

unsigned heat3d_invoke_ocl_naive (unsigned nb_iter)
{
  size_t global[1] = {GPU_SIZE}; // global domain size for our calculation
  size_t local[1]  = {TILE};     // local domain size for our calculation
  cl_int err;

  ocl_acquire ();

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    err |=
        clSetKernelArg (compute_kernel, 2, sizeof (cl_mem), &neighbors_buffer);
    err |=
        clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &index_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 1, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp  = cur_buffer;
      cur_buffer  = next_buffer;
      next_buffer = tmp;
      if (do_display)
        mesh3d_switch_data_color_buffer (ctx[0]);
    }
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, NB_CELLS, 0,
                       easypap_gpu_lane (TASK_TYPE_COMPUTE));

  ocl_release ();

  return 0;
}