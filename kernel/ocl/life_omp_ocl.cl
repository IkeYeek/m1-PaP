#include "kernel/ocl/common.cl"
typedef unsigned cell_t;
__kernel void life_omp_ocl_ocl (__global cell_t *in, __global cell_t *out)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < get_global_size(1) - 1) {
    const cell_t me = in[y * DIM + x];

    const unsigned n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                       in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                       in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                       in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a life_omp_ocl-specific version (generic version is defined in common.cl)
__kernel void life_omp_ocl_update_texture (__global cell_t *cur,
                                       __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  write_imagef (tex, (int2)(x, y),
                color_to_float4 (cur[y * DIM + x] * rgb (255, 255, 0)));
}
