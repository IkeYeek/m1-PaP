#include <stdbool.h>
#include <unistd.h>

#include "error.h"
#include "mon_obj.h"

void mon_obj_init (mon_obj_t *mon, unsigned cpu, unsigned gpu)
{
  mon->cpu = cpu;
  mon->gpu = gpu;
}

unsigned mon_obj_size (mon_obj_t *mon)
{
  return (mon->cpu + mon->gpu) * sizeof (unsigned);
}
