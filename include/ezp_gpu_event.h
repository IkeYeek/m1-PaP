#ifndef EZP_GPU_EVENT_H
#define EZP_GPU_EVENT_H

#ifdef ENABLE_OPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#include "easypap.h"
}
#else
#include "easypap.h"
#endif

typedef enum
{
  EVENT_START_KERNEL,
  EVENT_END_KERNEL,
  EVENT_START_TRANSFER0,
  EVENT_END_TRANSFER0,
  EVENT_START_TRANSFER1,
  EVENT_END_TRANSFER1,
  EVENT_START_KERNEL0,
  EVENT_END_KERNEL0,
  EVENT_START_KERNEL1,
  EVENT_END_KERNEL1,
  _EVENT_NB
} ezp_gpu_event_t;

typedef struct
{
  unsigned x, y, w, h;
} ezp_gpu_event_footprint_t;

void ezp_gpu_event_init (void);

// CUDA
#ifdef ENABLE_CUDA
void ezp_cuda_event_record (ezp_gpu_event_t evt, unsigned g);
void ezp_cuda_event_always_record (ezp_gpu_event_t evt, unsigned g);

void ezp_cuda_event_start (ezp_gpu_event_t evt, unsigned g);
void ezp_cuda_event_end (ezp_gpu_event_t evt, unsigned g);

void ezp_cuda_event_start_force (ezp_gpu_event_t evt, unsigned g);
void ezp_cuda_event_end_force (ezp_gpu_event_t evt, unsigned g);
#endif

#ifdef ENABLE_OPENCL
cl_event *ezp_ocl_eventptr (ezp_gpu_event_t evt, unsigned gpu);
#endif

void ezp_gpu_event_reset (void);
uint64_t ezp_gpu_event_monitor (int gpu, ezp_gpu_event_t start_evt,
                                uint64_t clock,
                                ezp_gpu_event_footprint_t *footp,
                                task_type_t task_type, int task_id);

void ezp_gpu_wait_event (int gpu_wait, int gpu_signal, ezp_gpu_event_t evt);

#define EZP_NO_FOOTPRINT NULL

#endif
