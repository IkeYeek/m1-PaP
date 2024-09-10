#include "kernel/ocl/common.cl"


__kernel void sample_ocl (__global unsigned *img)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  // x = amount of red, y = amount of blue
  unsigned color = rgb (x, 0, y);

  img [y * DIM + x] = color;
}
