#ifndef MESH_MGPU_OCL_H
#define MESH_MGPU_OCL_H

#include "gpu.h"
#include "ocl.h"

void mesh_mgpu_alloc_device_buffer (int gpu, cl_mem *buf, size_t size);

void mesh_mgpu_copy_host_to_device (int gpu, cl_mem dest_buffer, void *src_addr,
                                    size_t bytes, size_t offset_in_bytes);
void mesh_mgpu_copy_device_to_host (int gpu, void *dest_addr, cl_mem src_buffer,
                                    size_t bytes, size_t offset_in_bytes);

cl_kernel create_cell_gathering_kernel (void);

void mesh_mgpu_launch_cell_gathering_kernel (cl_kernel kernel, int gpu,
                                   const size_t threads, const size_t block,
                                   cl_mem arg0_curbuf, cl_mem arg1_outindex,
                                   cl_mem arg2_outval, unsigned arg3_outsize);

cl_mem mesh_mgpu_cur_buffer (int gpu);
cl_mem mesh_mgpu_get_soa_buffer (int gpu);

#endif
