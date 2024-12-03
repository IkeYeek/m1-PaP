
#include "cuda_kernels.h"
#include "debug.h"
#include "error.h"
#include "mesh_mgpu_cuda.h"
#include "minmax.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

EXTERN float *mesh_mgpu_cur_buffer (int gpu)
{
  return cuda_cur_data (gpu);
}

EXTERN void mesh_mgpu_alloc_device_buffer (int gpu, void **buf, size_t size)
{
  cudaError_t ret;

  ret = cudaSetDevice (cuda_device (gpu));
  check (ret, "cudaSetDevice");

  ret = cudaMalloc (buf, size);
  check (ret, "cudaMalloc");
}

EXTERN void mesh_mgpu_copy_host_to_device (int gpu, void *dest_buffer,
                                           void *src_addr, size_t bytes,
                                           size_t offset_in_bytes)
{
  cudaError_t ret;

  cudaSetDevice (cuda_device (gpu));
  ret = cudaMemcpyAsync ((char *)dest_buffer + offset_in_bytes, src_addr, bytes,
                         cudaMemcpyHostToDevice, cuda_stream (gpu));
  check (ret, "cudaMemcpyAsync");
  cudaStreamSynchronize (cuda_stream (gpu));
}

EXTERN void mesh_mgpu_copy_device_to_host (int gpu, void *dest_addr,
                                           void *src_buffer, size_t bytes,
                                           size_t offset_in_bytes)
{
  cudaError_t ret;

  cudaSetDevice (cuda_device (gpu));
  ret = cudaMemcpyAsync (dest_addr, (char *)src_buffer + offset_in_bytes, bytes,
                         cudaMemcpyDeviceToHost, cuda_stream (gpu));
  check (ret, "cudaMemcpyAsync");
  cudaStreamSynchronize (cuda_stream (gpu));
}

EXTERN void mesh_gpu_copy_device_to_device (int gpu, void *dest_buffer,
                                            void *src_buffer, size_t bytes)
{
  cudaError_t ret;

  ezp_cuda_event_record (EVENT_START_TRANSFER, gpu);
  ret = cudaMemcpyAsync (dest_buffer,
                         src_buffer, bytes,
                         cudaMemcpyDeviceToDevice, cuda_stream (gpu));
  check (ret, "cudaMemcpyAsync");
  ezp_cuda_event_record (EVENT_END_TRANSFER, gpu);
}

static __global__ void cuda_gather_cells (float *in, unsigned *indexes,
                                          float *out, unsigned nb)
{
  int index = gpu_get_col ();

  if (index < nb)
    out[index] = in[indexes[index]];
}

EXTERN void mesh_mgpu_launch_cell_gathering_kernel (
    int kernel, int gpu, const size_t threads, const size_t block,
    float *arg0_curbuf, unsigned *arg1_outindex, float *arg2_outval,
    unsigned arg3_outsize)
{
  unsigned grid = threads / block;

  cudaSetDevice (cuda_device (gpu));

  ezp_cuda_event_record (EVENT_START_KERNEL0, gpu);
  cuda_gather_cells<<<grid, block, 0, cuda_stream (gpu)>>> (
      arg0_curbuf, arg1_outindex, arg2_outval, arg3_outsize);
  ezp_cuda_event_always_record (EVENT_END_KERNEL0, gpu);
}

EXTERN void mesh_mgpu_wait_gathering_kernel (int gpu_wait, int gpu_signal)
{
  ezp_cuda_wait_event (gpu_wait, gpu_signal, EVENT_END_KERNEL0);
}
