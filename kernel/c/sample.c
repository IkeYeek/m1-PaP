
#include "easypap.h"

// If defined, the initialization hook function is called quite early in the
// initialization process, after the size (DIM variable) of images is known.
// This function can typically spawn a team of threads, or allocated additionnal
// OpenCL buffers.
// A function named <kernel>_init_<variant> is searched first. If not found, a
// function <kernel>_init is searched in turn.
void sample_init (void)
{
  PRINT_DEBUG ('u', "Image size is %dx%d\n", DIM, DIM);
  PRINT_DEBUG ('u', "Tile size is %dx%d\n", TILE_W, TILE_H);
  PRINT_DEBUG ('u', "Press <SPACE> to pause/unpause, <ESC> to quit.\n");
}

// The image is a two-dimension array of size of DIM x DIM. Each pixel is of
// type 'unsigned' and store the color information following a RGBA layout (4
// bytes). Pixel at line 'l' and column 'c' in the current image can be accessed
// using cur_img (l, c).

int sample_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = ezv_rgb (255, 255, 0);

  return 0;
}

// The kernel returns 0, or the iteration step at which computation has
// completed (e.g. stabilized).

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run --size 1024 --kernel sample --variant seq
// or
// ./run -s 1024 -k sample -v seq
//
unsigned sample_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
    do_tile (0, 0, DIM, DIM);

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -k sample -v tiled -g 16 -m
// or
// ./run -k sample -v tiled -ts 64 -m
//
unsigned sample_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);
  }

  return 0;
}


#ifdef ENABLE_OPENCL

//////////////////////////////////////////////////////////////////////////
///////////////////////////// OpenCL version
// Suggested cmdlines:
// ./run -k sample -g
//
unsigned sample_compute_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X,
                      GPU_SIZE_Y};     // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                           &ocl_cur_buffer (0));

    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");
  }

  clFinish (ocl_queue (0));

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}

#endif
