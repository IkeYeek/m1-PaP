
#ifndef EASYPAP_NVIDIA_CUDA_H
#define EASYPAP_NVIDIA_CUDA_H

#include "cppdefs.h"

EXTERN unsigned easypap_number_of_gpus_cuda (void);

EXTERN void cuda_init (int show_config, int silent);
EXTERN void cuda_alloc_buffers (void);
EXTERN void cuda_build_program (int list_variants);

EXTERN void cuda_send_data (void);
EXTERN void cuda_retrieve_data (void);

EXTERN unsigned cuda_compute (unsigned nb_iter);

EXTERN void cuda_update_texture (void);

EXTERN void cuda_establish_bindings (void);

EXTERN void *cuda_alloc_host (size_t size);
EXTERN void cuda_free_host (void *ptr);

#endif // EASYPAP_NVIDIA_CUDA_H
