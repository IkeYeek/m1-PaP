#ifndef EASYPAP_GPU_H
#define EASYPAP_GPU_H

#include "hooks.h"

#define MAX_NB_GPU 2

#ifdef ENABLE_OPENCL ////////// OPENCL //////////

#include "ocl.h"
#include "debug.h"

#define GPU_CAN_BE_USED 1

#define DEFAULT_GPU_VARIANT DEFAULT_OCL_VARIANT

#define easypap_number_of_gpus() easypap_number_of_gpus_ocl ()

#define gpu_init(b1, b2) ocl_init (b1, b2)

#define gpu_build_program(b) ocl_build_program (b)

#define gpu_alloc_buffers() ocl_alloc_buffers ()

#define gpu_send_data() ocl_send_data ()

#define gpu_retrieve_data() ocl_retrieve_data ()

#define gpu_update_texture() ocl_update_texture ()

#define gpu_establish_bindings() ocl_establish_bindings ()


#elif defined(ENABLE_CUDA) ////////// CUDA //////////

#include "nvidia_cuda.h"

#define GPU_CAN_BE_USED 1

#define DEFAULT_GPU_VARIANT DEFAULT_CUDA_VARIANT

#define easypap_number_of_gpus() easypap_number_of_gpus_cuda ()

#define gpu_init(b1, b2) cuda_init (b1, b2)

#define gpu_build_program(b) cuda_build_program (b)

#define gpu_alloc_buffers() cuda_alloc_buffers ()

#define gpu_send_data() cuda_send_data ()

#define gpu_retrieve_data() cuda_retrieve_data ()

#define gpu_update_texture() cuda_update_texture ()

#define gpu_establish_bindings() cuda_establish_bindings ()


#else ////////// NO GPU //////////

#define GPU_CAN_BE_USED 0

#define DEFAULT_GPU_VARIANT NULL

#define easypap_number_of_gpus() 0

#define gpu_init(b1, b2)

#define gpu_build_program(b)

#define gpu_alloc_buffers()

#define gpu_send_data()

#define gpu_retrieve_data()

#define gpu_map_textures(id)

#define gpu_update_texture()

#define gpu_establish_bindings()


#endif

#endif // EASYPAP_GPU_H