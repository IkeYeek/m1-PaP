#include "kernel/ocl/common.cl"
typedef unsigned cell_t;
__kernel void life_ocl (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile_in[TILE_H+2][TILE_W+2];
  __local unsigned neigh_count[TILE_H][TILE_W];
  unsigned x         = get_global_id (0);
  unsigned y         = get_global_id (1);
  unsigned xloc_true = get_local_id(0);
  unsigned yloc_true = get_local_id(1);
  unsigned xloc      = xloc_true + 1;  // Fixed: using xloc_true for initialization
  unsigned yloc      = yloc_true + 1;  // Fixed: using yloc_true for initialization
  unsigned global_id = y * DIM + x;
  
  // first we load all the values from global memory to local tile memory
  tile_in[yloc][xloc] = in[y*DIM+x];
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // then we let the first thread of the warp load the borders
  if (xloc_true == 0) {
    if (yloc_true == 0) {
      if (y == 0) {
        tile_in[yloc-1][xloc-1] = 0;
      } else if (x == 0) {
        tile_in[yloc-1][xloc-1] = 0;
      } else {
        tile_in[yloc-1][xloc-1] = in[(y-1) * DIM + (x-1)];
      }
    } else if (yloc_true == TILE_H-1) {
      if (y == DIM-1) {
        tile_in[yloc+1][xloc-1] = 0;
      } else if (x == 0) {
        tile_in[yloc+1][xloc-1] = 0;
      } else {
        tile_in[yloc+1][xloc-1] = in[(y+1) * DIM + (x-1)];
      }
    }
    if (x == 0) {
      tile_in[yloc][xloc-1] = 0;
    } else {
      tile_in[yloc][xloc-1] = in[y * DIM + (x-1)];
    }
  } else if (xloc_true == TILE_W - 1) {
    if (yloc_true == 0) {
      if (y == 0) {
        tile_in[yloc-1][xloc+1] = 0;
      } else if (x == DIM-1) {
        tile_in[yloc-1][xloc+1] = 0;
      } else {
        tile_in[yloc-1][xloc+1] = in[(y-1) * DIM + (x+1)];
      }
    } else if (yloc_true == TILE_H-1) {
      if (y == DIM-1) {
        tile_in[yloc+1][xloc+1] = 0;
      } else if (x == DIM-1) {
        tile_in[yloc+1][xloc+1] = 0;
      } else {
        tile_in[yloc+1][xloc+1] = in[(y+1)*DIM+(x+1)];
      }
    }
    if (x == DIM - 1) {
      tile_in[yloc][xloc+1] = 0;
    } else {
      tile_in[yloc][xloc+1] = in[y * DIM + (x+1)];
    }
  }
  
  if (yloc_true == 0) {
    if (y == 0) {
      tile_in[yloc-1][xloc] = 0;
    } else {
      tile_in[yloc-1][xloc] = in[(y-1) * DIM + x];
    }
  } else if (yloc_true == TILE_H - 1) {
    if (y == DIM - 1) {
      tile_in[yloc+1][xloc] = 0;
    } else {
      tile_in[yloc+1][xloc] = in[(y+1) * DIM + x];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // finally we load back our tile into global memory
  out[global_id] = tile_in[yloc][xloc];
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
