#include <fcntl.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ENABLE_FUT

#define CONFIG_FUT
#include <fut.h>
#define BUFFER_SIZE (16 << 20)

#endif

#include "error.h"
#include "trace_common.h"
#include "trace_data.h"
#include "trace_record.h"

unsigned do_trace          = 0;
unsigned trace_may_be_used = 0;

static unsigned cache_activated = 0;

static unsigned task_ids_count = 0;
static unsigned NB_CPUS        = 0;

void trace_record_init (char *file, unsigned cpu, unsigned gpu, unsigned dim,
                        char *label, unsigned starting_iteration,
                        unsigned is_cache_enabled)
{
  NB_CPUS = cpu;

  fut_set_filename (file);
  enable_fut_flush ();

  if (fut_setup (BUFFER_SIZE, 0xffff, 0) < 0)
    exit_with_error ("fut_setup");

  // We use 2 lanes per GPU : one for computations, the other for data transfers
  FUT_DO_PROBE2 (TRACE_NB_THREADS, cpu, gpu * 2);
  FUT_DO_PROBE1 (TRACE_DIM, dim);
  if (label != NULL)
    FUT_DO_PROBESTR (TRACE_LABEL, label);
  FUT_DO_PROBE1 (TRACE_FIRST_ITER, starting_iteration);
  FUT_DO_PROBE1 (TRACE_DO_CACHE, is_cache_enabled);

  cache_activated = is_cache_enabled;
}

void trace_record_set_meshfile (char *filename)
{
  if (filename != NULL)
    FUT_DO_PROBESTR (TRACE_MESHFILE, filename);
}

void trace_record_set_palette (int palette)
{
  FUT_DO_PROBE1 (TRACE_PALETTE, palette);
}

void trace_record_finalize (void)
{
  if (fut_endup ("temp") < 0)
    exit_with_error ("fut_endup");

  if (fut_done () < 0)
    exit_with_error ("fut_done");
}

void trace_record_declare_task_ids (char *task_ids[])
{
  if (!trace_may_be_used)
    return;

  task_ids_count = 1;

  if (task_ids != NULL) {
    for (int i = 0; task_ids[i] != NULL; i++) {
      task_ids_count++;
    }
  }

  FUT_DO_PROBE1 (TRACE_TASKID_COUNT, task_ids_count);
  FUT_DO_PROBESTR (TRACE_TASKID, "anonymous"); // task id 0
  if (task_ids != NULL)
    for (int i = 0; task_ids[i] != NULL; i++)
      FUT_DO_PROBESTR (TRACE_TASKID, task_ids[i]); // task id i + 1
}

void trace_record_commit_task_ids (void)
{
  if (task_ids_count == 0) // declare_task_ids not called by user
    trace_record_declare_task_ids (NULL);
}

void __trace_record_start_iteration ()
{
  FUT_DO_PROBE0 (TRACE_BEGIN_ITER);
}

void __trace_record_end_iteration ()
{
  FUT_DO_PROBE0 (TRACE_END_ITER);
}

static int cpu_to_lane (int cpu, task_type_t task_type)
{
  if (cpu < NB_CPUS)
    return cpu;

  return NB_CPUS + 2 * (cpu - NB_CPUS) +
         (task_type == TASK_TYPE_COMPUTE ? 0 : 1);
}

void __trace_record_start_tile (long time, unsigned cpu, task_type_t task_type)
{
  FUT_DO_PROBE2 (TRACE_BEGIN_TILE, time, cpu_to_lane (cpu, task_type));
}

void __trace_record_end_tile (long time, unsigned cpu, unsigned x, unsigned y,
                              unsigned w, unsigned h, int task_type,
                              int task_id, int64_t *counters)
{
  if (task_id >= task_ids_count)
    exit_with_error (
        "monitoring_end_tile: task id %d is too large (should < %d)%s\n",
        task_id, task_ids_count,
        (task_ids_count == 1)
            ? ". Probable cause: monitoring_declare_task_ids not called"
            : "");

  if (cache_activated)
    FUT_DO_PROBE9 (TRACE_END_TILE, time, cpu_to_lane (cpu, task_type), x, y, w,
                   h, INT_COMBINE (task_type, task_id), counters[0],
                   counters[1]);
  else
    FUT_DO_PROBE7 (TRACE_END_TILE, time, cpu_to_lane (cpu, task_type), x, y, w,
                   h, INT_COMBINE (task_type, task_id));
}

void __trace_record_tile (long start_time, unsigned cpu, unsigned x, unsigned y,
                          unsigned w, unsigned h, int task_type, int task_id,
                          int64_t *counters)
{
  if (task_id >= task_ids_count)
    exit_with_error (
        "monitoring_end_tile: task id %d is too large (should < %d)%s\n",
        task_id, task_ids_count,
        (task_ids_count == 1)
            ? ". Probable cause: monitoring_declare_task_ids not called"
            : "");

  if (cache_activated)
    FUT_DO_PROBE7 (TRACE_TILE, start_time, cpu_to_lane (cpu, task_type),
                   INT_COMBINE (x, y), INT_COMBINE (w, h),
                   INT_COMBINE (task_type, task_id), counters[0], counters[1]);
  else
    FUT_DO_PROBE5 (TRACE_TILE, start_time, cpu_to_lane (cpu, task_type),
                   INT_COMBINE (x, y), INT_COMBINE (w, h),
                   INT_COMBINE (task_type, task_id));
}
