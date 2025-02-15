#ifndef COMMON3D_IS_DEF
#define COMMON3D_IS_DEF
//
// !!! DO NOT MODIFY THIS FILE !!!
//
// Utility functions for OpenCL
//

#ifndef NB_CELLS
#define NB_CELLS 1
#define MAX_NEIGHBORS 1
#define GPU_SIZE 1
#define TILE 1
#endif


__kernel void bench_kernel (void)
{
}

__kernel void gather_outgoing_cells (__global float *in, __global unsigned *indexes, __global float *out, unsigned nb)
{
  const unsigned index = get_global_id (0);

  if (index < nb)
    out[index] = in[indexes[index]];
}

#endif