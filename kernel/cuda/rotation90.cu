#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

EXTERN __global__ void rotation90_cuda (unsigned *in, unsigned *out,
                                        unsigned DIM)
{
  unsigned i = gpu_get_row ();
  unsigned j = gpu_get_col ();

  out[(DIM - i - 1) * DIM + j] = in[j * DIM + i];
}