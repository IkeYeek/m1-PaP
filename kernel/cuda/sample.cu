#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

// Suggested cmdline:
//   ./run -k sample -g -i 1
static __global__ void sample_cuda (unsigned *img, unsigned DIM)
{
  unsigned x = gpu_get_col ();
  unsigned y = gpu_get_row ();

  img[y * DIM + x] = rgb (255, 255, 0);
}

// We redefine the kernel launcher function because
// the kernel onlu uses a single image
EXTERN unsigned sample_compute_cuda (unsigned nb_iter)
{
  cudaError_t ret;
  dim3 grid  = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / TILE_H, 1};
  dim3 block = {TILE_W, TILE_H, 1};

  cudaSetDevice (cuda_device (0));

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (int i = 0; i < nb_iter; i++)
    sample_cuda<<<grid, block, 0, cuda_stream (0)>>> (cuda_cur_buffer (0), DIM);

  // FIXME: should only be performed when monitoring/tracing is activated
  ret = cudaStreamSynchronize (cuda_stream (0));
  check (ret, "cudaStreamSynchronize");

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}
