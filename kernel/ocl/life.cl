#include "kernel/ocl/common.cl"
#include <opencl-c-base.h>
__kernel void life_ocl (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[TILE_H][TILE_W];
  unsigned x         = get_global_id (0);
  unsigned y         = get_global_id (1);
  unsigned xloc      = get_local_id (0);
  unsigned yloc      = get_local_id (1);
  unsigned global_id = y * DIM + x;
  tile[yloc][xloc]   = in[global_id];

  barrier (CLK_LOCAL_MEM_FENCE);

  tile[yloc][xloc] = 1;

  barrier (CLK_LOCAL_MEM_FENCE);
  out[global_id] = tile[yloc][xloc];
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a life-specific version (generic version is defined in common.cl)
__kernel void life_update_texture (__global unsigned *cur,
                                   __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);

  write_imagef (tex, (int2)(x, y),
                color_to_float4 (cur[y * DIM + x] * rgb (255, 255, 0)));
}
