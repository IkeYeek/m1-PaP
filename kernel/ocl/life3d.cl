#include <opencl-c-base.h>
__kernel void life3d_ocl_naive (__global float *in, __global float *out,
                                __global int *neighbors,
                                __global int *index_neighbor)
{
  const int index = get_global_id (0);

  if (index < NB_CELLS) {
    int cur = index;

    int me = in[cur];

    int top       = neighbors[index_neighbor[cur]];
    int top_right = neighbors[index_neighbor[top] + 1];
    int top_left  = neighbors[index_neighbor[top] + 3];

    int right = neighbors[index_neighbor[cur] + 1];
    int left  = neighbors[index_neighbor[cur] + 3];

    int bottom       = neighbors[index_neighbor[cur] + 2];
    int bottom_right = neighbors[index_neighbor[bottom] + 1];
    int bottom_left  = neighbors[index_neighbor[bottom] + 3];

    int n = in[top] + in[top_right] + in[top_left] + in[right] + in[left] +
            in[bottom] + in[bottom_right] + in[bottom_left];

    const char new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
    out[cur]          = new_me;
  }
}

__kernel void life3d_ocl (__global float *in, __global float *out,
                          __global int *neighbor_soa)
{
  const int index = get_global_id (0);
  out[index]      = 0;
  if (index < NB_CELLS) {

    int me = in[index];

    int top       = neighbor_soa[index];
    int top_right = neighbor_soa[1 * NB_CELLS + top];
    int top_left  = neighbor_soa[3 * NB_CELLS + top];

    int right = neighbor_soa[1 * NB_CELLS + index];
    int left  = neighbor_soa[3 * NB_CELLS + index];

    int bottom       = neighbor_soa[2 * NB_CELLS + index];
    int bottom_right = neighbor_soa[1 * NB_CELLS + bottom];
    int bottom_left  = neighbor_soa[3 * NB_CELLS + bottom];

    int n = in[top] + in[top_right] + in[top_left] + in[right] + in[left] +
            in[bottom] + in[bottom_right] + in[bottom_left];

    out[index] = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
  }
}
static inline bool __in (int a, int b, int c)
{
  return c >= a && c <= b;
}
__kernel void life3d_ocl_cache (__global float *in, __global float *out,
                                __global int *neighbor_soa)
{
  const int index       = get_global_id (0);
  const int group_index = get_group_id (0);
  const int group_start = group_index * TILE;
  const int group_end   = group_start + TILE - 1;
  const int loc_index   = get_local_id (0);
  __local float tile[TILE];
  if (index < NB_CELLS) {
    tile[loc_index] = in[index];
    barrier (CLK_LOCAL_MEM_FENCE);
    int me = in[index];

    int top       = neighbor_soa[index];
    int top_right = neighbor_soa[1 * NB_CELLS + top];
    int top_left  = neighbor_soa[3 * NB_CELLS + top];

    int right = neighbor_soa[1 * NB_CELLS + index];
    int left  = neighbor_soa[3 * NB_CELLS + index];

    int bottom       = neighbor_soa[2 * NB_CELLS + index];
    int bottom_right = neighbor_soa[1 * NB_CELLS + bottom];
    int bottom_left  = neighbor_soa[3 * NB_CELLS + bottom];
    int n;
    if (__in (group_start, group_end, top_left) &&
        __in (group_start, group_end, top_right) &&
        __in (group_start, group_end, bottom_left) &&
        __in (group_start, group_end, bottom_right)) {
      n = tile[top - group_start] + tile[top_right - group_start] +
          tile[top_left - group_start] + tile[right - group_start] +
          tile[left - group_start] + tile[bottom - group_start] +
          tile[bottom_right - group_start] + tile[bottom_left - group_start];

    } else {
      n = in[top] + in[top_right] + in[top_left] + in[right] + in[left] +
          in[bottom] + in[bottom_right] + in[bottom_left];
    }

    out[index] = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
  }
}
