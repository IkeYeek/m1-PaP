#include "kernel/ocl/common3d.cl"

__kernel void sample3d_ocl (__global float *img)
{
  const int index = get_global_id (0);

  if (index < NB_CELLS)
    img [index] = (float)index / (float)(NB_CELLS - 1);
}
