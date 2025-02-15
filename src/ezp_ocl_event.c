#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "ezp_gpu_event.h"

static cl_event the_events[2][_EVENT_NB];

static uint32_t recorded_events = 0;
static uint64_t ref_clock[2];

static ezp_gpu_event_footprint_t zero_footprint = {0, 0, 0, 0};

static inline void mark_recorded (ezp_gpu_event_t evt, unsigned gpu)
{
  recorded_events |= 1U << (gpu * _EVENT_NB + evt);
}

static inline unsigned is_recorded (ezp_gpu_event_t evt, unsigned gpu)
{
  return (recorded_events & (1U << (gpu * _EVENT_NB + evt))) != 0;
}

void ezp_gpu_event_init (void)
{
  ezp_gpu_event_reset ();
}

void ezp_gpu_event_reset (void)
{
  // Free events
  for (int gpu = 0; gpu < 2; gpu++)
    for (int evt = 0; evt < _EVENT_NB; evt++)
      if (is_recorded ((ezp_gpu_event_t)evt, gpu))
        clReleaseEvent (the_events[gpu][evt]);

  recorded_events = 0;
  ref_clock[0]    = 0;
  ref_clock[1]    = 0;
}

cl_event *ezp_ocl_eventptr (ezp_gpu_event_t evt, unsigned gpu)
{
  mark_recorded (evt, gpu);
  return &the_events[gpu][evt];
}

void ezp_gpu_wait_event (int gpu_wait, int gpu_signal, ezp_gpu_event_t evt)
{
  if (is_recorded (evt, gpu_signal))
    exit_with_error ("OpenCL version of gpu_wait_event not yet implemented");
  else
    exit_with_error ("Cannot wait for event %d from GPU %d (not recorded)\n",
                     evt, gpu_signal);
}

uint64_t ezp_gpu_event_monitor (int gpu, ezp_gpu_event_t evt, uint64_t clock,
                                ezp_gpu_event_footprint_t *footp,
                                task_type_t task_type, int task_id)
{
  uint64_t stamp[2]; // wallclock time

  if (is_recorded (evt, gpu)) {
    cl_ulong start, end; // OpenCL timestamps

    // start
    clGetEventProfilingInfo (the_events[gpu][evt], CL_PROFILING_COMMAND_START,
                             sizeof (cl_ulong), &start, NULL);

    clGetEventProfilingInfo (the_events[gpu][evt], CL_PROFILING_COMMAND_END,
                             sizeof (cl_ulong), &end, NULL);

    if (ref_clock[gpu] == 0) {
      // This is the first event to be recorded
      ref_clock[gpu] = start;
      stamp[0] = clock;
    } else {
      stamp[0] = (start - ref_clock[gpu]) / 1000 + clock;
    }

    stamp[1] = (end - ref_clock[gpu]) / 1000 + clock;

    if (footp == EZP_NO_FOOTPRINT)
      footp = &zero_footprint;

    if (footp->h)
      monitoring_gpu_tile (footp->x, footp->y, footp->w, footp->h,
                           easypap_gpu_lane (gpu), stamp[0], stamp[1],
                           task_type, task_id);
    else
      monitoring_gpu_patch (footp->x, footp->w, easypap_gpu_lane (gpu),
                            stamp[0], stamp[1], task_type, task_id);

    return stamp[1] - stamp[0];
  }
  
  return 0;
}
