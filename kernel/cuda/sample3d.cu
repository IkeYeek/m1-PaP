#include "cppdefs.h"
#include "cuda_kernels.h"
EXTERN
{
#include "easypap.h"
}

EXTERN __global__ void sample3d_cuda (float *in, float *out, int *neighbor_soa, unsigned nb_cells, unsigned max_neighbors)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < nb_cells) {
    out [index] = (float)index / (float)(nb_cells - 1);
  }
}
