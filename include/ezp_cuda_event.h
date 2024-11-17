#ifndef EZP_CUDA_EVENT_H
#define EZP_CUDA_EVENT_H

#ifdef __cplusplus
extern "C"
{
#include "easypap.h"
}
#else
#include "easypap.h"
#endif

typedef enum
{
  EVENT_START_KERNEL,
  EVENT_END_KERNEL,
  EVENT_START_TRANSFER,
  EVENT_END_TRANSFER,
  EVENT_START_KERNEL0,
  EVENT_END_KERNEL0,
  _EVENT_NB
} ezp_cuda_event_t;

typedef struct
{
  unsigned x, y, w, h;
} ezp_cuda_event_footprint_t;

void ezp_cuda_event_init (char **taskids);
void ezp_cuda_event_record (ezp_cuda_event_t evt, unsigned g);
void ezp_cuda_event_reset (void);
void ezp_cuda_event_monitor (int gpu, ezp_cuda_event_t start_evt,
                             uint64_t clock, ezp_cuda_event_footprint_t *footp,
                             task_type_t task_type, int task_id);

void ezp_cuda_wait_event (int gpu_wait, int gpu_signal, ezp_cuda_event_t evt);

#define EZP_NO_FOOTPRINT  NULL

#endif
