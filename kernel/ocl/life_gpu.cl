#include "kernel/ocl/common.cl"
typedef unsigned cell_t;
__kernel void life_gpu_ocl (__global cell_t *in, __global cell_t *out)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
    const cell_t me = in[y * DIM + x];

    const unsigned n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                       in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                       in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                       in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}

__kernel void life_gpu_ocl_binmul (__global cell_t *in, __global cell_t *out,
                                   const unsigned shift_by)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
    const cell_t me = in[(y << shift_by) + x];

    const unsigned n =
        in[((y - 1) << shift_by) + (x - 1)] + in[((y - 1) << shift_by) + x] +
        in[((y - 1) << shift_by) + (x + 1)] + in[((y << shift_by) + (x - 1))] +
        in[((y << shift_by) + (x + 1))] + in[((y + 1) << shift_by) + (x - 1)] +
        in[((y + 1) << shift_by) + x] + in[((y + 1) << shift_by) + (x + 1)];
    const cell_t new_me      = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[(y << shift_by) + x] = new_me;
  }
}

__kernel void life_gpu_ocl_2x (__global cell_t *in, __global cell_t *out)
{
  const int x  = get_global_id (0);
  const int x2 = x + get_global_size (0);
  const int y  = get_global_id (1);
  cell_t new_me;
  cell_t new_me2;
  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
    const cell_t me = in[y * DIM + x];

    const int n = in[(y - 1) * DIM + (x - 1)] + in[(y - 1) * DIM + x] +
                  in[(y - 1) * DIM + (x + 1)] + in[(y * DIM + (x - 1))] +
                  in[(y * DIM + (x + 1))] + in[(y + 1) * DIM + (x - 1)] +
                  in[(y + 1) * DIM + x] + in[(y + 1) * DIM + (x + 1)];

    new_me           = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x] = new_me;
  }
  if (x2 > 0 && x2 < DIM - 1 && y > 0 && y < DIM - 1) {
    const cell_t me2 = in[y * DIM + x2];
    const int n2     = in[(y - 1) * DIM + (x2 - 1)] + in[(y - 1) * DIM + x2] +
                   in[(y - 1) * DIM + (x2 + 1)] + in[(y * DIM + (x2 - 1))] +
                   in[(y * DIM + (x2 + 1))] + in[(y + 1) * DIM + (x2 - 1)] +
                   in[(y + 1) * DIM + x2] + in[(y + 1) * DIM + (x2 + 1)];
    new_me2           = (me2 & ((n2 == 2) | (n2 == 3))) | (!me2 & (n2 == 3));
    out[y * DIM + x2] = new_me2;
  }
}

__kernel void life_gpu_ocl_more_explicit_vec (__global cell_t *in,
                                              __global cell_t *out)
{
  const unsigned x = get_global_id (0);
  const unsigned y = get_global_id (1);

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
    const cell_t me = in[y * DIM + x];

    const uint3 line_above =
        (uint3)(in[(y - 1) * DIM + (x - 1)], in[(y - 1) * DIM + x],
                in[(y - 1) * DIM + (x + 1)]);
    const uint2 line_cell =
        (uint2)(in[(y * DIM + (x - 1))], in[(y * DIM + (x + 1))]);
    const uint3 line_below =
        (uint3)(in[(y + 1) * DIM + (x - 1)], in[(y + 1) * DIM + x],
                in[(y + 1) * DIM + (x + 1)]);

    uint3 identity_3 = (uint3)(1, 1, 1);
    uint2 identity_2 = (uint2)(1, 1);
    const unsigned n = line_above.x + line_above.y + line_above.z +
                       line_cell.x + line_cell.y + line_below.x + line_below.y +
                       line_below.z;
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}

__kernel void life_gpu_ocl_localmem (__global cell_t *in, __global cell_t *out)
{
  __local cell_t TILE[TILE_H + 2][TILE_W + 2];
  const unsigned x       = get_global_id (0);
  const unsigned y       = get_global_id (1);
  const unsigned local_x = get_local_id (0);
  const unsigned local_y = get_local_id (1);
  const unsigned tile_x  = local_x + 1;
  const unsigned tile_y  = local_y + 1;

  // load center cell
  if (x < DIM && y < DIM)
    TILE[tile_y][tile_x] = in[y * DIM + x];
  // load halo cells
  if (local_x < 1 && x > 0) {
    TILE[tile_y][0] = in[y * DIM + (x - 1)];
  }

  if (local_x >= TILE_W - 1 && x < DIM - 1) {
    // right edge
    TILE[tile_y][TILE_W] = in[y * DIM + (x + 1)];
  }

  if (local_y < 1 && y > 0) {
    // top edge
    TILE[0][tile_x] = in[(y - 1) * DIM + x];

    // top
    if (local_x < 1 && x > 0)
      TILE[0][0] = in[(y - 1) * DIM + (x - 1)];

    if (local_x >= TILE_W - 1 && x < DIM - 1)
      TILE[0][TILE_W] = in[(y - 1) * DIM + (x + 1)];
  }

  if (local_y >= TILE_W - 1 && y < DIM - 1) {
    // bottom edge
    TILE[TILE_H][tile_x] = in[(y + 1) * DIM + x];

    // bottom corners
    if (local_x < 1 && x > 0)
      TILE[TILE_H][0] = in[(y + 1) * DIM + (x - 1)];

    if (local_x >= TILE_W - 1 && x < DIM - 1)
      TILE[TILE_H][TILE_W] = in[(y + 1) * DIM + (x + 1)];
  }
  barrier (CLK_LOCAL_MEM_FENCE);
  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {
    const cell_t me = TILE[tile_y][tile_x];

    const unsigned n =
        TILE[(tile_y - 1)][(tile_x - 1)] + TILE[(tile_y - 1)][tile_x] +
        TILE[(tile_y - 1)][(tile_x + 1)] + TILE[(tile_y)][(tile_x - 1)] +
        TILE[(tile_y)][(tile_x + 1)] + TILE[(tile_y + 1)][(tile_x - 1)] +
        TILE[(tile_y + 1)][tile_x] + TILE[(tile_y + 1)][(tile_x + 1)];
    const cell_t new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[y * DIM + x]    = new_me;
  }
}

// pretty much same strategy as the CPU one to check if it is as efficient
__kernel void life_gpu_ocl_lazy (__global cell_t *in, __global cell_t *out,
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
    tile_out[tile_idx] = 0;
    tile_change        = 0;
    compute_tile       = tile_in[tile_idx];
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
  if (yloc == 0 && xloc == 0 && tile_change) {
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
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a life_gpu-specific version (generic version is defined in common.cl)
__kernel void life_gpu_update_texture (__global cell_t *cur,
                                       __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  write_imagef (tex, (int2)(x, y),
                color_to_float4 (cur[y * DIM + x] * rgb (255, 255, 0)));
}
