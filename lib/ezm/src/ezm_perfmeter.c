#include <stdlib.h>
#include <unistd.h>

#include "error.h"
#include "ezm_perfmeter.h"
#include "ezm_time.h"
#include "ezv.h"
#include "ezv_virtual.h"

struct ezm_perfm_struct
{
  unsigned nb_pus;
  uint64_t it_start, it_end;
  float *activity_ratio;
  struct
  {
    uint64_t start_time;
    uint64_t work_time;
    uint64_t overhead;
  } *cpu_stat;
  ezv_ctx_t ezv_ctx;
};

static float compute_activity_ratio (ezm_perfmeter_t rec, int who)
{
  uint64_t work = rec->cpu_stat[who].work_time;

  if (work == 0)
    return 0.0;

  uint64_t total = (rec->it_end - rec->it_start) - rec->cpu_stat[who].overhead;

  return (float)work / (float)total;
}

ezm_perfmeter_t ezm_perfmeter_create (unsigned nb_cpus, unsigned nb_gpus,
                                      ezv_ctx_t ctx)
{
  if (ezv_ctx_type (ctx) != EZV_CTX_TYPE_MONITOR)
    exit_with_error ("Invalid context type (%s)", ezv_ctx_typestr (ctx));

  if (ezv_get_color_data_size (ctx) != nb_cpus + nb_gpus)
    exit_with_error (
        "Inconsistency between #PUs (%d) and ctx configuration (%d)",
        nb_cpus + nb_gpus, ezv_get_color_data_size (ctx));

  ezm_perfmeter_t rec = (ezm_perfmeter_t)malloc (sizeof (*rec));

  rec->nb_pus   = nb_cpus + nb_gpus;
  rec->it_start = 0;
  rec->it_end   = 0;
  rec->ezv_ctx  = ctx;

  rec->activity_ratio = malloc ((rec->nb_pus) * sizeof (float));
  rec->cpu_stat       = malloc ((rec->nb_pus) * sizeof (*rec->cpu_stat));

  return rec;
}

void ezm_perfmeter_enable (ezm_perfmeter_t rec)
{
  ezv_ctx_show (rec->ezv_ctx);
}

void ezm_perfmeter_disable (ezm_perfmeter_t rec)
{
  ezv_ctx_hide (rec->ezv_ctx);
}

void ezm_perfmeter_it_start (ezm_perfmeter_t rec, uint64_t now)
{
  memset (rec->cpu_stat, 0, rec->nb_pus * sizeof (*rec->cpu_stat));

  rec->it_start = now;
}

void ezm_perfmeter_it_end (ezm_perfmeter_t rec, uint64_t now)
{
  rec->it_end = now;

  for (int p = 0; p < rec->nb_pus; p++)
    rec->activity_ratio[p] = compute_activity_ratio (rec, p);

  ezv_set_data_colors (rec->ezv_ctx, rec->activity_ratio);
}

void ezm_perfmeter_start_work (ezm_perfmeter_t rec, uint64_t now, int who)
{
  rec->cpu_stat[who].start_time = now;
}

uint64_t ezm_perfmeter_finish_work (ezm_perfmeter_t rec, uint64_t now, int who)
{
  uint64_t duration = now - rec->cpu_stat[who].start_time;

  // How long did the cpu work?
  rec->cpu_stat[who].work_time += duration;

  return duration;
}

void ezm_perfmeter_substract_overhead (ezm_perfmeter_t rec, uint64_t duration,
                                     int who)
{
  rec->cpu_stat[who].overhead += duration;
}

void ezm_perfmeter_destroy (ezm_perfmeter_t rec)
{
  free (rec->activity_ratio);
  free (rec->cpu_stat);
  free (rec);
}
