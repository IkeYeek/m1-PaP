
#include "easypap.h"

#include <omp.h>

static unsigned LOG_BLOC = 0; // LOG2(TILE_W^2)

static unsigned log2_of_power_of_2 (unsigned v)
{
  const unsigned b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
  const unsigned S[] = {1, 2, 4, 8, 16};

  unsigned r = 0;

  for (int i = 4; i >= 0; i--)
    if (v & b[i]) {
      v >>= S[i];
      r |= S[i];
    }

  return r;
}

// The parameter is used to fix the size of pixelized blocks
void pixelize_config (char *param)
{
  if (TILE_W != TILE_H)
    exit_with_error ("Tiles should have a square shape (%d != %d)", TILE_W,
                     TILE_H);

  LOG_BLOC = 2 * log2_of_power_of_2 (TILE_W);
}

// Tile computation
int pixelize_do_tile_default (int x, int y, int width, int height)
{
  uint32_t r = 0, g = 0, b = 0, a = 0, mean;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) {
      uint32_t c = cur_img (i, j);
      r += ezv_c2r (c);
      g += ezv_c2g (c);
      b += ezv_c2b (c);
      a += ezv_c2a (c);
    }

  // Divide by TILE_W^2 (i.e. shift right by 2*log2(TILE_W) bits)
  r >>= LOG_BLOC;
  g >>= LOG_BLOC;
  b >>= LOG_BLOC;
  a >>= LOG_BLOC;

  mean = ezv_rgba (r, g, b, a);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = mean;

  return 0;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run -l data/img/1024.png -k pixelize -ts 16
//
unsigned pixelize_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);
  }

  return 0;
}

#ifdef ENABLE_OPENCL

///////////////////////////// OpenCL variant

unsigned pixelize_compute_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local[2]  = {TILE_W, TILE_H};
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


unsigned pixelize_compute_ocl_fake (unsigned nb_iter)
{
  return pixelize_compute_ocl (nb_iter);
}

unsigned pixelize_compute_ocl_big (unsigned nb_iter)
{
  return pixelize_compute_ocl (nb_iter);
}

unsigned pixelize_compute_ocl_1D (unsigned nb_iter)
{
  return pixelize_compute_ocl (nb_iter);
}

#endif
