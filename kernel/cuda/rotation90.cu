#include "cuda_kernels.cuh"
#include "cppdefs.h"
EXTERN {
#include "easypap.h"
}

EXTERN __global__ void rotation90_cuda(unsigned *in, unsigned *out, unsigned DIM) {
  unsigned i = get_i ();
  unsigned j = get_j ();
  out[(DIM - i - 1) * DIM + j] = in[j * DIM + i];
}