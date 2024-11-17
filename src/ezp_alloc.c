#include "ezp_alloc.h"
#include "error.h"
#include "global.h"

#ifdef ENABLE_CUDA
#include "nvidia_cuda.h"
#endif

#include <sys/mman.h>

void *ezp_alloc (size_t size)
{
#ifdef ENABLE_CUDA
  if (gpu_used) {
    return cuda_alloc_host (size);
  } else
#endif
  {
    void *ptr = NULL;

    ptr = mmap (NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
    if (ptr == NULL)
      exit_with_error ("Cannot allocate memory: mmap failed");

    return ptr;
  }
}

void ezp_free (void *ptr, size_t size)
{
#ifdef ENABLE_CUDA
  if (gpu_used) {
    cuda_free_host (ptr);
  } else
#endif
  {
    munmap (ptr, size);
  }
}
