#include "kernel/ocl/common.cl"
typedef char cell_t;
__kernel void life_omp_ocl_ocl_hybrid (__global cell_t *in, __global cell_t *out)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < get_global_size (1) - 1) {
    const cell_t me = in[y * DIM + x];

    const unsigned n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                       in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                       in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                       in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}

__kernel void life_omp_ocl_ocl_mt (__global cell_t *in, __global cell_t *out)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < get_global_size (1) - 1) {
    const cell_t me = in[y * DIM + x];

    const unsigned n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                       in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                       in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                       in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}
//l
__kernel void life_omp_ocl_ocl_hybrid_dyn (__global cell_t *in,
                                         __global cell_t *out)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < get_global_size (1) - 1) {
    const cell_t me = in[y * DIM + x];

    const unsigned n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                       in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                       in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                       in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}

__kernel void life_omp_ocl_ocl_hybrid_conv (__global cell_t *in,
                                         __global cell_t *out)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < get_global_size (1) - 1) {
    const cell_t me = in[y * DIM + x];

    const unsigned n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                       in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                       in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                       in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}

__kernel void life_omp_ocl_ocl_hybrid_lazy (__global cell_t *in, __global cell_t *out,
                                 __global cell_t *tile_in,
                                 __global cell_t *tile_out)
{
  unsigned x          = get_global_id (0);
  unsigned y          = get_global_id (1);
  unsigned xloc       = get_local_id (0);
  unsigned yloc       = get_local_id (1);
  unsigned xgroup     = get_group_id (0);
  unsigned ygroup     = get_group_id (1);
  unsigned NB_TILES_W = DIM / TILE_W;
  unsigned xtile      = xgroup + 1;
  unsigned ytile      = ygroup + 1;
  unsigned tile_idx   = ytile * NB_TILES_W + xtile;
  __local unsigned compute_tile;
  __local unsigned tile_change;
  // first of the warp (so first of the tile as well)
  if (xloc == 0 && yloc == 0) {
    // tile_out[tile_idx] = 0;
    tile_change  = 0;
    compute_tile = tile_in[tile_idx];
  }
  barrier (CLK_LOCAL_MEM_FENCE);
  if (!compute_tile)
    return;
  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
    const cell_t me  = in[y * DIM + x];
    const unsigned n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                       in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                       in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                       in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    if (new_me != me) {
      tile_change = true;
    }
    out[y * DIM + x] = new_me;
  }
  barrier (CLK_LOCAL_MEM_FENCE);
  if (yloc == 0 && xloc == 0) {
    if (tile_change) {
      tile_out[tile_idx]                               = 1;
      tile_out[ytile * NB_TILES_W + (xtile + 1)]       = 1;
      tile_out[(ytile - 1) * NB_TILES_W + (xtile - 1)] = 1;
      tile_out[(ytile - 1) * NB_TILES_W + xtile]       = 1;
      tile_out[(ytile - 1) * NB_TILES_W + (xtile + 1)] = 1;
      tile_out[ytile * NB_TILES_W + (xtile - 1)]       = 1;
      tile_out[(ytile + 1) * NB_TILES_W + (xtile - 1)] = 1;
      tile_out[(ytile + 1) * NB_TILES_W + xtile]       = 1;
      tile_out[(ytile + 1) * NB_TILES_W + (xtile + 1)] = 1;
    }
    tile_in[tile_idx] = 0;
  }
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a life_omp_ocl-specific version (generic version is defined in
// common.cl)
__kernel void life_omp_ocl_update_texture (__global cell_t *cur,
                                           __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  write_imagef (tex, (int2)(x, y),
                color_to_float4 (cur[y * DIM + x] * rgb (255, 255, 0)));
}
