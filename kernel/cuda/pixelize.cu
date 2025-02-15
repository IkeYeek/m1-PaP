#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

EXTERN __global__ void pixelize_cuda_fake (uint32_t *img,
                                           unsigned DIM)
{
  unsigned index = gpu_get_index ();
  __shared__ uint32_t color;

  if (threadIdx.x + threadIdx.y == 0)
    color = img[index];

  __syncthreads ();

  img[index] = color;
}

__device__ static int4 int4_add (int4 a, int4 b)
{
  return make_int4 (a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ static int4 int4_div_int (int4 a, int b)
{
  return make_int4 (a.x / b, a.y / b, a.z / b, a.w / b);
}

// We redefine the kernel launcher function because
// we use shared memory of variable length
EXTERN unsigned pixelize_compute_cuda_fake (unsigned nb_iter)
{
  cudaError_t ret;
  dim3 grid  = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / TILE_H, 1};
  dim3 block = {TILE_W, TILE_H, 1};

  ret = cudaSetDevice (cuda_device (0));
  check (ret, "cudaSetDevice");

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (int i = 0; i < nb_iter; i++)
    pixelize_cuda<<<grid, block, 0,
                    cuda_stream (0)>>> (cuda_cur_buffer (0), DIM);

  ret = cudaStreamSynchronize (cuda_stream (0));
  check (ret, "cudaStreamSynchronize");

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}
