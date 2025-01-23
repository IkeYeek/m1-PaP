#include <stdlib.h>
#include <unistd.h>

#include "error.h"
#include "ezm_time.h"
#include "ezm_tracerec.h"
#include "ezm_pack.h"

#ifdef ENABLE_FUT
#define CONFIG_FUT
#include <fut.h>
#define BUFFER_SIZE (16 << 20)
#endif

struct ezm_trrec_struct
{
  unsigned nb_cpus;
  unsigned task_ids_count;
  struct
  {
    uint64_t start_time;
  } *pu_stat;
};

static int cpu_to_lane (ezm_tracerec_t rec, int cpu, task_type_t task_type)
{
  if (cpu < rec->nb_cpus)
    return cpu;

  return rec->nb_cpus + 2 * (cpu - rec->nb_cpus) +
         (task_type == TASK_TYPE_COMPUTE ? 0 : 1);
}

ezm_tracerec_t ezm_tracerec_create (unsigned nb_cpus, unsigned nb_gpus,
                                    char *file, char *label)
{
  ezm_tracerec_t rec = (ezm_tracerec_t)malloc (sizeof (*rec));

  rec->nb_cpus        = nb_cpus;
  rec->task_ids_count = 0;

  rec->pu_stat = malloc ((nb_cpus + 2 * nb_gpus) * sizeof (*rec->pu_stat));

  fut_set_filename (file);
  enable_fut_flush ();

  if (fut_setup (BUFFER_SIZE, 0xffff, 0) < 0)
    exit_with_error ("fut_setup");

  // We use 2 lanes per GPU : one for computations, the other for data transfers
  FUT_DO_PROBE2 (TRACE_NB_THREADS, nb_cpus, nb_gpus * 2);
  if (label != NULL)
    FUT_DO_PROBESTR (TRACE_LABEL, label);

  return rec;
}

void ezm_tracerec_enable (ezm_tracerec_t rec, unsigned iter)
{
  FUT_DO_PROBE1 (TRACE_FIRST_ITER, iter);
}

void ezm_tracerec_disable (ezm_tracerec_t rec)
{
}

void ezm_tracerec_store_mesh3d_filename (ezm_tracerec_t rec,
                                         const char *filename)
{
  if (filename != NULL)
    FUT_DO_PROBESTR (TRACE_MESHFILE, filename);
}

void ezm_tracerec_store_img2d_dim (ezm_tracerec_t rec, unsigned width,
                                   unsigned height)
{
  FUT_DO_PROBE2 (TRACE_DIM, width, height);
}

void ezm_tracerec_store_data_palette (ezm_tracerec_t rec, ezv_palette_name_t pal)
{
  FUT_DO_PROBE1 (TRACE_PALETTE, pal);
}

void ezm_tracerec_declare_task_ids (ezm_tracerec_t rec, char *task_ids[])
{
  if (task_ids != NULL) {
    for (int i = 0; task_ids[i] != NULL; i++) {
      rec->task_ids_count++;
    }
  }

  FUT_DO_PROBE1 (TRACE_TASKID_COUNT, rec->task_ids_count + 1);
  FUT_DO_PROBESTR (TRACE_TASKID, "anonymous"); // task id 0 won't be used
  if (task_ids != NULL)
    for (int i = 0; task_ids[i] != NULL; i++)
      FUT_DO_PROBESTR (TRACE_TASKID, task_ids[i]); // task id i + 1
}

void ezm_tracerec_it_start (ezm_tracerec_t rec)
{
  FUT_DO_PROBE0 (TRACE_BEGIN_ITER);
}

void ezm_tracerec_it_end (ezm_tracerec_t rec)
{
  FUT_DO_PROBE0 (TRACE_END_ITER);
}

void ezm_tracerec_start_work (ezm_tracerec_t rec, int cpu)
{
  rec->pu_stat[cpu].start_time = ezm_gettime ();
}

#define shift_id(task_id) (task_id + (rec->task_ids_count ? 1 : 0))


// 1D patches (i.e. meshes)

// Short version
void ezm_tracerec_1D (ezm_tracerec_t rec, int cpu, unsigned patch,
                      unsigned count)
{
  FUT_DO_PROBE3 (TRACE_PATCH_MIN, rec->pu_stat[cpu].start_time,
                 cpu_to_lane (rec, cpu, TASK_TYPE_COMPUTE),
                 INT_COMBINE (patch, count));
}

// Full version (with task type)
void ezm_tracerec_1D_task (ezm_tracerec_t rec, int cpu, unsigned patch,
                           unsigned count, int task_id)
{
  FUT_DO_PROBE4 (TRACE_PATCH, rec->pu_stat[cpu].start_time,
                 cpu_to_lane (rec, cpu, TASK_TYPE_COMPUTE), INT_COMBINE (patch, count),
                 INT_COMBINE (TASK_TYPE_COMPUTE, shift_id (task_id)));
}

// Extended version (with start/end time)
void ezm_tracerec_1D_ext (ezm_tracerec_t rec, uint64_t start_time,
                               uint64_t end_time, int cpu, unsigned patch,
                               unsigned count, task_type_t task_type,
                               int task_id)
{
  FUT_DO_PROBE5 (TRACE_PATCH_EXT, start_time, cpu_to_lane (rec, cpu, task_type),
                 end_time, INT_COMBINE (patch, count),
                 INT_COMBINE (task_type, shift_id (task_id)));
}

// 2D tiles (i.e. images)

// Short version
void ezm_tracerec_2D (ezm_tracerec_t rec, int cpu, unsigned x, unsigned y,
                      unsigned w, unsigned h)
{
  FUT_DO_PROBE4 (TRACE_TILE_MIN, rec->pu_stat[cpu].start_time,
                 cpu_to_lane (rec, cpu, TASK_TYPE_COMPUTE), INT_COMBINE (x, y),
                 INT_COMBINE (w, h));
}

// Full version (with task type)
void ezm_tracerec_2D_task (ezm_tracerec_t rec, int cpu, unsigned x, unsigned y,
                           unsigned w, unsigned h, int task_id)
{
  FUT_DO_PROBE5 (TRACE_TILE, rec->pu_stat[cpu].start_time,
                 cpu_to_lane (rec, cpu, TASK_TYPE_COMPUTE), INT_COMBINE (x, y),
                 INT_COMBINE (w, h), INT_COMBINE (TASK_TYPE_COMPUTE, shift_id (task_id)));
}

// Extended version (with start/end time)
void ezm_tracerec_2D_ext (ezm_tracerec_t rec, uint64_t start_time,
                               uint64_t end_time, int cpu, unsigned x,
                               unsigned y, unsigned w, unsigned h,
                               task_type_t task_type, int task_id)
{
  FUT_DO_PROBE6 (TRACE_TILE_EXT, start_time, cpu_to_lane (rec, cpu, task_type),
                 end_time, INT_COMBINE (x, y), INT_COMBINE (w, h),
                 INT_COMBINE (task_type, shift_id (task_id)));
}

void ezm_tracerec_destroy (ezm_tracerec_t rec)
{
  if (fut_endup ("temp") < 0)
    exit_with_error ("fut_endup");

  if (fut_done () < 0)
    exit_with_error ("fut_done");

  free (rec->pu_stat);
  free (rec);
}
