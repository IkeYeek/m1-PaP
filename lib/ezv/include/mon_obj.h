#ifndef MON_OBJ_H
#define MON_OBJ_H


typedef struct
{
  unsigned cpu;
  unsigned gpu;
} mon_obj_t;

void mon_obj_init (mon_obj_t *mon, unsigned cpu, unsigned gpu);

unsigned mon_obj_size (mon_obj_t *mon);


#endif
