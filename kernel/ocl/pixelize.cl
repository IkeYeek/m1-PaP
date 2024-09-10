#include "kernel/ocl/common.cl"

#ifdef PARAM
#define PIX_BLOC PARAM
#else
#define PIX_BLOC 16
#endif

// In this over-simplified kernel, all the pixels of a bloc adopt the color
// on the top-left pixel (i.e. we do not compute the average color).
__kernel void pixelize_ocl_fake (__global unsigned *in)
{
  __local unsigned couleur;
  int x    = get_global_id (0);
  int y    = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  if (xloc == 0 && yloc == 0)
    couleur = in[y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  in[y * DIM + x] = couleur;
}

__kernel void pixelize_ocl (__global unsigned *in)
{
  // TODO
}

__kernel void pixelize_ocl_1D (__global unsigned *in)
{
  // TODO
}
