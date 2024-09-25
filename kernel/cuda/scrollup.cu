#include "cppdefs.h"
#include "cuda_kernels.cuh"
EXTERN
{
#include "easypap.h"
}

EXTERN __global__ void scrollup_cuda (unsigned *in, unsigned *out, unsigned DIM)
{
  unsigned i = get_i ();
  unsigned j = get_j ();

  unsigned i2 = (i < DIM - 1) ? i + 1 : 0;

  out[i * DIM + j] = in[i2 * DIM + j];
}