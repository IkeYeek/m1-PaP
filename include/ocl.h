#ifndef OCL_IS_DEF
#define OCL_IS_DEF

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#include "error.h"
#include "monitoring.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


void ocl_init (int show_config, int silent);
void ocl_build_program (int list_variants);
void ocl_alloc_buffers (void);
void ocl_map_textures (int texid);
void ocl_send_data (void);
void ocl_retrieve_data (void);
unsigned ocl_invoke_kernel_generic (unsigned nb_iter);
void ocl_update_texture (void);
unsigned easypap_number_of_gpus_ocl (void);
size_t ocl_get_max_workgroup_size (void);
const char *ocl_GetError (cl_int error);

#define check(err, format, ...)                                                \
  do {                                                                         \
    if (err != CL_SUCCESS)                                                     \
      exit_with_error (format " [OCL err %d]", ##__VA_ARGS__, err);            \
  } while (0)

extern cl_context context;
extern cl_program program;
extern cl_kernel compute_kernel;
extern cl_command_queue queue;
extern cl_mem cur_buffer, next_buffer;

int64_t ocl_monitor (cl_event evt, int x, int y, int width, int height,
                  task_type_t task_type);

void ocl_acquire (void);
void ocl_release (void);

#endif
