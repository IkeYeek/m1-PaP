#include "cuda_kernels.h"

static cudaEvent_t the_events[2][_EVENT_NB];

static uint32_t recorded_events = 0;

static ezp_cuda_event_footprint_t zero_footprint = {0, 0, 0, 0};

static inline void mark_recorded (ezp_cuda_event_t evt, unsigned gpu)
{
  recorded_events |= 1U << (gpu * _EVENT_NB + evt);
}

static inline unsigned is_recorded (ezp_cuda_event_t evt, unsigned gpu)
{
  return (recorded_events & (1U << (gpu * _EVENT_NB + evt))) != 0;
}

static void create_events (void)
{
  // always create events
  cudaError_t ret;

  for (int g = 0; g < cuda_nb_gpus; g++) {
    cudaSetDevice (cuda_device (g));
    for (int e = 0; e < _EVENT_NB; e++) {
      ret = cudaEventCreate (&the_events[g][e]);
      check (ret, "cudaEventCreate");
    }
  }
}

void ezp_cuda_event_init (char **taskids)
{
  create_events ();

  if (taskids != NULL)
    monitoring_declare_task_ids (taskids);
}

void ezp_cuda_event_reset (void)
{
  recorded_events = 0;
}

void ezp_cuda_event_record (ezp_cuda_event_t evt, unsigned g)
{
  if (ezp_monitoring_is_active ()) {
    cudaError_t ret;

    ret = cudaEventRecord (the_events[g][evt], cuda_stream (g));
    check (ret, "cudaEventRecord(ev:%d)", evt);

    mark_recorded (evt, g);
  }
}

void ezp_cuda_event_always_record (ezp_cuda_event_t evt, unsigned g)
{
  cudaError_t ret;

  ret = cudaEventRecord (the_events[g][evt], cuda_stream (g));
  check (ret, "cudaEventRecord(ev:%d)", evt);

  mark_recorded (evt, g);
}

void ezp_cuda_wait_event (int gpu_wait, int gpu_signal, ezp_cuda_event_t evt)
{
  if (is_recorded (evt, gpu_signal))
    cudaStreamWaitEvent (cuda_stream (gpu_wait), the_events[gpu_signal][evt]);
  else
    exit_with_error ("Cannot wait for event %d from GPU %d (not recorded)\n",
                     evt, gpu_signal);
}

void ezp_cuda_event_monitor (int gpu, ezp_cuda_event_t start_evt,
                             uint64_t clock, ezp_cuda_event_footprint_t *footp,
                             task_type_t task_type, int task_id)
{
  uint64_t stamp[2]; // start + end
  float time_ms;

  if (is_recorded (start_evt, gpu) &&
      is_recorded ((ezp_cuda_event_t)((int)start_evt + 1), gpu)) {
    // start
    cudaEventElapsedTime (&time_ms, the_events[gpu][start_evt],
                          the_events[gpu][EVENT_END_KERNEL]);
    stamp[0] = clock - (uint64_t)(time_ms * 1000.0);
    // end
    cudaEventElapsedTime (
        &time_ms, the_events[gpu][(ezp_cuda_event_t)((int)start_evt + 1)],
        the_events[gpu][EVENT_END_KERNEL]);
    stamp[1] = clock - (uint64_t)(time_ms * 1000.0);

    if (footp == EZP_NO_FOOTPRINT)
      footp = &zero_footprint;

    if (footp->h)
      monitoring_gpu_tile (footp->x, footp->y, footp->w, footp->h,
                           easypap_gpu_lane (gpu), stamp[0], stamp[1],
                           task_type, task_id);
    else
      monitoring_gpu_patch (footp->x, footp->w, easypap_gpu_lane (gpu),
                            stamp[0], stamp[1], task_type, task_id);
  }
}
