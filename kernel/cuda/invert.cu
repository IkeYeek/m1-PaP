#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

EXTERN __global__ void invert_cuda (unsigned *in, unsigned *out, unsigned DIM)
{
  unsigned index = gpu_get_index ();

  out[index] = in[index] ^ rgb_mask ();
}