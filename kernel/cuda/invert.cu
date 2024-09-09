#include "cuda_kernels.cuh"
#include "cppdefs.h"
EXTERN {
#include "easypap.h"
}

EXTERN __global__ void invert_cuda(unsigned *in, unsigned *out, unsigned DIM) {
  unsigned index = get_index ();
  out[index] = in[index] ^ rgb_invert_mask();
}