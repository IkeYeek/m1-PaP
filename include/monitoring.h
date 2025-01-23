#ifndef MONITORING_IS_DEF
#define MONITORING_IS_DEF

#include "ezm.h"
#include "perfcounter.h"
#include "time_macros.h"

extern ezm_recorder_t ezp_monitor;
extern char easypap_trace_label[];
extern unsigned do_trace;
extern unsigned trace_may_be_used;
extern unsigned do_gmonitor;

void ezp_monitoring_init (unsigned nb_cpus, unsigned nb_gpus);
void ezp_monitoring_cleanup (void);

static inline int ezp_monitoring_is_active (void)
{
  return ezm_recorder_is_enabled (ezp_monitor);
}

static inline void monitoring_declare_task_ids (char *task_ids[])
{
  ezm_recorder_declare_task_ids (ezp_monitor, task_ids);
}

static inline void monitoring_start_iteration (void)
{
  ezm_start_iteration (ezp_monitor);
}

static inline void monitoring_end_iteration (void)
{
  ezm_end_iteration (ezp_monitor);
}

static inline void monitoring_start (unsigned cpu)
{
  ezm_start_work (ezp_monitor, cpu);
}

// 1D

static inline void monitoring_end_patch (unsigned patch, unsigned count,
                                         unsigned cpu)
{
  ezm_end_1D (ezp_monitor, cpu, patch, count);
}

static inline void monitoring_end_patch_id (unsigned patch, unsigned count,
                                            unsigned cpu, unsigned task_id)
{
  ezm_end_1D_task (ezp_monitor, cpu, patch, count, task_id);
}

static inline void monitoring_gpu_patch (unsigned patch, unsigned count,
                                         unsigned cpu, long start, long end,
                                         task_type_t task_type,
                                         unsigned task_id)
{
  ezm_1D_ext (ezp_monitor, start, end, cpu, patch, count, task_type, task_id);
}

// 2D

static inline void monitoring_end_tile (unsigned x, unsigned y, unsigned w,
                                        unsigned h, unsigned cpu)
{
  ezm_end_2D (ezp_monitor, cpu, x, y, w, h);
}

static inline void monitoring_end_tile_id (unsigned x, unsigned y, unsigned w,
                                           unsigned h, unsigned cpu,
                                           unsigned task_id)
{
  ezm_end_2D_task (ezp_monitor, cpu, x, y, w, h, task_id);
}

static inline void monitoring_gpu_tile (unsigned x, unsigned y, unsigned w,
                                        unsigned h, unsigned cpu, long start,
                                        long end, task_type_t task_type,
                                        unsigned task_id)
{
  ezm_2D_ext (ezp_monitor, start, end, cpu, x, y, w, h, task_type, task_id);
}

#endif
