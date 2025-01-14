#ifndef EZV_MON_H
#define EZV_MON_H

#include "mon_obj.h"

struct ezv_ctx_s;
typedef struct ezv_ctx_s *ezv_ctx_t;

void ezv_mon_set_moninfo (ezv_ctx_t ctx, mon_obj_t *moninfo);

// For private use only
void ezv_mon_get_suggested_window_size (mon_obj_t *mon, unsigned *width,
                                        unsigned *height);

#endif
