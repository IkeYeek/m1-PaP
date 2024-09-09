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
void ocl_send_data (void);
void ocl_retrieve_data (void);
unsigned ocl_compute (unsigned nb_iter);
void ocl_establish_bindings (void);
void ocl_update_texture (void);
unsigned easypap_number_of_gpus_ocl (void);
size_t ocl_get_max_workgroup_size (void);
const char *ocl_GetError (cl_int error);

#define check(err, format, ...)                                                \
  do {                                                                         \
    if (err != CL_SUCCESS)                                                     \
      exit_with_error (format " [OCL err %d: %s]", ##__VA_ARGS__, err,         \
                       ocl_GetError (err));                                    \
  } while (0)

typedef struct
{
  cl_command_queue q;
  cl_device_id device;
  int64_t calibration_delta;
  cl_mem curb, nextb;
  cl_kernel kernel;
} ocl_gpu_t;

extern ocl_gpu_t ocl_gpu[];
extern unsigned ocl_nb_gpus;

extern cl_context context;
extern cl_program program;

#define queue ocl_gpu[0].q
#define cur_buffer ocl_gpu[0].curb
#define next_buffer ocl_gpu[0].nextb
#define compute_kernel ocl_gpu[0].kernel

int64_t ocl_monitor (cl_event evt, int x, int y, int width, int height,
                     task_type_t task_type, unsigned gpu_no);

void ocl_acquire (void);
void ocl_release (void);

typedef struct
{
  int64_t start, end;
} ocl_stamp_t;

void ocl_link_stamp (cl_event evt, ocl_stamp_t *stamp);

#endif
