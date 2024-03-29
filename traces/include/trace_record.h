#ifndef TRACE_RECORD_IS_DEF
#define TRACE_RECORD_IS_DEF

#include "trace_common.h"

#ifdef ENABLE_TRACE

extern unsigned do_trace;
extern unsigned trace_may_be_used;

void trace_record_init (char *file, unsigned cpu, unsigned gpu, unsigned dim,
                        char *label, unsigned starting_iteration,
                        unsigned cache);
void trace_record_set_meshfile (char *filename);
void trace_record_set_palette (int palette);
void trace_record_declare_task_ids (char *task_ids[]);
void trace_record_commit_task_ids (void);
void __trace_record_start_iteration ();
void __trace_record_end_iteration ();
void __trace_record_start_tile (long time, unsigned cpu);
void __trace_record_end_tile (long time, unsigned cpu, unsigned x, unsigned y,
                              unsigned w, unsigned h, int task_type,
                              int task_id, int64_t *counters);
void __trace_record_tile (long start_time, unsigned cpu, unsigned x, unsigned y,
                          unsigned w, unsigned h, int task_type, int task_id,
                          int64_t *counters);
void trace_record_finalize (void);

#define trace_record_start_iteration()                                         \
  do {                                                                         \
    if (do_trace)                                                              \
      __trace_record_start_iteration ();                                       \
  } while (0)

#define trace_record_end_iteration()                                           \
  do {                                                                         \
    if (do_trace)                                                              \
      __trace_record_end_iteration ();                                         \
  } while (0)

#define trace_record_start_tile(t, c)                                          \
  do {                                                                         \
    if (do_trace)                                                              \
      __trace_record_start_tile ((t), (c));                                    \
  } while (0)

#define trace_record_end_tile(t, c, x, y, w, h, tt, tid, counters)             \
  do {                                                                         \
    if (do_trace)                                                              \
      __trace_record_end_tile ((t), (c), (x), (y), (w), (h), (tt), (tid),      \
                               (counters));                                    \
  } while (0)

#define trace_record_tile(t, c, x, y, w, h, tt, tid, counters)                 \
  do {                                                                         \
    if (do_trace)                                                              \
      __trace_record_tile ((t), (c), (x), (y), (w), (h), (tt), (tid),          \
                           (counters));                                        \
  } while (0)

#else

#define do_trace (unsigned)0
#define trace_may_be_used (unsigned)0

#define trace_record_declare_task_ids(a) (void)0
#define trace_record_start_iteration() (void)0
#define trace_record_end_iteration() (void)0
#define trace_record_start_tile(t, c) (void)0
#define trace_record_end_tile(t, c, x, y, w, h, tt, tid, counters) (void)0
#define trace_record_tile(t, c, x, y, w, h, tt, tid, counters) (void)0

#endif

#endif
