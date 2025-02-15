#include "kernel/ocl/common3d.cl"

__kernel void heat3d_ocl_naive (__global float *in, __global float *out, __global int *neighbors, __global int *index_neighbor)
{
  const int index = get_global_id (0);

  if (index < NB_CELLS) {
    // TODO
  }
}

__kernel void heat3d_ocl (__global float *in, __global float *out, __global int *neighbor_soa)
{
  const int index = get_global_id (0);

  if (index < NB_CELLS) {
    // TODO
  }
}
