#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

EXTERN __global__ void scrollup_cuda (unsigned *in, unsigned *out, unsigned DIM)
{
  unsigned i = gpu_get_row ();
  unsigned j = gpu_get_col ();

  unsigned i2 = (i < DIM - 1) ? i + 1 : 0;

  out[i * DIM + j] = in[i2 * DIM + j];
}
