#ifndef MONITORING_IS_DEF
#define MONITORING_IS_DEF

#include "gmonitor.h"
#include "perfcounter.h"
#include "time_macros.h"
#include "trace_record.h"

#ifdef ENABLE_MONITORING

static inline void monitoring_declare_task_ids (char *task_ids[])
{
  trace_record_declare_task_ids (task_ids);
}

static inline void monitoring_start_iteration (void)
{
#ifdef ENABLE_PAPI
  if (do_cache)
    easypap_perfcounter_monitor_start_iteration ();
#endif
  if (do_gmonitor)
    gmonitor_start_iteration (what_time_is_it ());
  if (do_trace)
    trace_record_start_iteration ();
}

static inline void monitoring_end_iteration (void)
{
#ifdef ENABLE_PAPI
  if (do_cache)
    easypap_perfcounter_monitor_stop_iteration ();
#endif
  if (do_gmonitor)
    gmonitor_end_iteration (what_time_is_it ());
  if (do_trace)
    trace_record_end_iteration ();
}

static inline uint64_t monitoring_start_tile (unsigned cpu)
{
  uint64_t t = 0;
  if (do_gmonitor | do_trace)
    t = what_time_is_it ();

#ifdef ENABLE_PAPI
  if (do_cache)
    easypap_perfcounter_monitor_start_tile (cpu);
#endif

  return t;
}

static inline void monitoring_end_tile (uint64_t clock, unsigned x, unsigned y,
                                        unsigned w, unsigned h, unsigned cpu)
{
#ifdef ENABLE_PAPI
  if (do_cache)
    easypap_perfcounter_monitor_stop_tile (cpu);
#endif

  if (do_gmonitor)
    gmonitor_tile (clock, what_time_is_it (), cpu, x, y, w, h);
  if (do_trace) {
#ifdef ENABLE_PAPI
    if (do_cache) {
      int64_t counters[EASYPAP_NB_COUNTERS];
      easypap_perfcounter_get_counters (counters, cpu);
      trace_record_tile (clock, cpu, x, y, w, h, TASK_TYPE_COMPUTE, 0,
                         counters);
    } else {
      trace_record_tile (clock, cpu, x, y, w, h, TASK_TYPE_COMPUTE, 0, NULL);
    }
#else
    trace_record_tile (clock, cpu, x, y, w, h, TASK_TYPE_COMPUTE, 0, NULL);
#endif
  }
}

static inline void monitoring_end_tile_id (uint64_t clock, unsigned x,
                                           unsigned y, unsigned w, unsigned h,
                                           unsigned cpu, unsigned task_id)
{
#ifdef ENABLE_PAPI
  if (do_cache)
    easypap_perfcounter_monitor_stop_tile (cpu);
#endif

  if (do_gmonitor)
    gmonitor_tile (clock, what_time_is_it (), cpu, x, y, w, h);
  if (do_trace) {
#ifdef ENABLE_PAPI
    if (do_cache) {
      int64_t counters[EASYPAP_NB_COUNTERS];
      easypap_perfcounter_get_counters (counters, cpu);
      trace_record_tile (clock, cpu, x, y, w, h, TASK_TYPE_COMPUTE, task_id + 1,
                         counters);
    } else {

      trace_record_tile (clock, cpu, x, y, w, h, TASK_TYPE_COMPUTE, task_id + 1,
                         NULL);
    }
#else
    trace_record_tile (clock, cpu, x, y, w, h, TASK_TYPE_COMPUTE, task_id + 1,
                       NULL);
#endif
  }
}

static inline void monitoring_gpu_tile (unsigned x, unsigned y, unsigned w,
                                        unsigned h, unsigned cpu, long start,
                                        long end, task_type_t task_type)
{
  if (do_gmonitor | do_trace) {
    trace_record_start_tile (start, cpu, task_type);
    if (task_type == TASK_TYPE_COMPUTE)
      gmonitor_tile (start, end, cpu, x, y, w, h);
    trace_record_end_tile (end, cpu, x, y, w, h, task_type, 0, NULL);
  }
}

#else

#define monitoring_declare_task_ids (task_ids) (void) 0
#define monitoring_start_iteration() (void)0
#define monitoring_end_iteration() (void)0
#define monitoring_start_tile(c) (void)0
#define monitoring_end_tile(cl, x, y, w, h, c) (void)0
#define monitoring_end_tile_id(cl, x, y, w, h, c, id) (void)0
#define monitoring_gpu_tile(x, y, w, h, c, s, e, tt) (void)0

#endif

#endif
