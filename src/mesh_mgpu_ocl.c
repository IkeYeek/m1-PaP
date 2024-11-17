
#include "mesh_mgpu_ocl.h"

void mesh_mgpu_alloc_device_buffer (int gpu, cl_mem *buf, size_t size)
{
  cl_mem buffer = clCreateBuffer (context, CL_MEM_READ_ONLY, size, NULL, NULL);
  if (!buffer)
    exit_with_error ("Failed to allocate buffer\n");

  *buf = buffer;
}

void mesh_mgpu_copy_host_to_device (int gpu, cl_mem dest_buffer, void *src_addr,
                                    size_t bytes, size_t offset_in_bytes)
{
  cl_int err =
      clEnqueueWriteBuffer (ocl_queue (gpu), dest_buffer, CL_TRUE,
                            offset_in_bytes, bytes, src_addr, 0, NULL, NULL);
  check (err, "Failed to write to device buffer");
}

void mesh_mgpu_copy_device_to_host (int gpu, void *dest_addr, cl_mem src_buffer,
                                    size_t bytes, size_t offset_in_bytes)
{
  cl_int err =
      clEnqueueReadBuffer (ocl_queue (gpu), src_buffer, CL_TRUE,
                           offset_in_bytes, bytes, dest_addr, 0, NULL, NULL);
  check (err, "Failed to read from device buffer");
}

cl_kernel create_cell_gathering_kernel (void)
{
  cl_int err;

  cl_kernel k = clCreateKernel (program, "gather_outgoing_cells", &err);
  check (err, "Failed to create 'gather_outgoing_buffer' kernel");

  return k;
}

void mesh_mgpu_launch_cell_gathering_kernel (cl_kernel kernel, int gpu,
                                   const size_t threads, const size_t block,
                                   cl_mem arg0_curbuf, cl_mem arg1_outindex,
                                   cl_mem arg2_outval, unsigned arg3_outsize)
{
  cl_int err;

  // Set gather kernel arguments
  //
  err = 0;
  err |= clSetKernelArg (kernel, 0, sizeof (cl_mem), &arg0_curbuf);
  err |= clSetKernelArg (kernel, 1, sizeof (cl_mem), &arg1_outindex);
  err |= clSetKernelArg (kernel, 2, sizeof (cl_mem), &arg2_outval);
  err |= clSetKernelArg (kernel, 3, sizeof (unsigned), &arg3_outsize);
  check (err, "Failed to set kernel arguments");

  err = clEnqueueNDRangeKernel (ocl_queue (gpu), kernel, 1, NULL, &threads,
                                &block, 0, NULL, NULL);
  check (err, "Failed to execute kernel");
}

cl_mem mesh_mgpu_cur_buffer (int gpu)
{
  return ocl_cur_buffer (gpu);
}
