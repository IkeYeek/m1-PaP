#ifndef MESH_MGPU_CUDA_H
#define MESH_MGPU_CUDA_H

#include "cppdefs.h"
#include <unistd.h>

EXTERN void mesh_mgpu_alloc_device_buffer (int gpu, void **buf, size_t size);

EXTERN void mesh_mgpu_copy_host_to_device (int gpu, void *dest_buffer,
                                           void *src_addr, size_t bytes,
                                           size_t offset_in_bytes);
EXTERN void mesh_mgpu_copy_device_to_host (int gpu, void *dest_addr,
                                           void *src_buffer, size_t bytes,
                                           size_t offset_in_bytes);
EXTERN void mesh_gpu_copy_device_to_device (int gpu, void *dest_buffer,
                                            void *src_buffer, size_t bytes);

#define create_cell_gathering_kernel() 0

EXTERN void mesh_mgpu_launch_cell_gathering_kernel (
    int kernel, int gpu, const size_t threads, const size_t block,
    float *arg0_curbuf, unsigned *arg1_outindex, float *arg2_outval,
    unsigned arg3_outsize);
EXTERN void mesh_mgpu_wait_gathering_kernel (int gpu_wait, int gpu_signal);

EXTERN float *mesh_mgpu_cur_buffer (int gpu);
EXTERN int *mesh_mgpu_get_soa_buffer (int gpu);

#endif
