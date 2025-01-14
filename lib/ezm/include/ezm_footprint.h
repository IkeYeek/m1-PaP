#ifndef EZM_FOOTPRINT_H
#define EZM_FOOTPRINT_H

#include "ezv.h"

#include <stdint.h>

struct ezm_footp_struct;
typedef struct ezm_footp_struct *ezm_footprint_t;

ezm_footprint_t ezm_footprint_create (unsigned nb_pus, ezv_palette_t *palette,
                                      unsigned cyclic_mode, ezv_ctx_t ctx);
void ezm_footprint_destroy (ezm_footprint_t rec);

void ezm_footprint_enable (ezm_footprint_t rec);
void ezm_footprint_disable (ezm_footprint_t rec);

void ezm_footprint_it_start (ezm_footprint_t rec, uint64_t now);
void ezm_footprint_it_end (ezm_footprint_t rec, uint64_t now);
void ezm_footprint_start_work (ezm_footprint_t rec, uint64_t now, int who);
void ezm_footprint_finish_work_1D (ezm_footprint_t rec, uint64_t now, int who,
                                   unsigned patch, unsigned count);
void ezm_footprint_finish_work_2D (ezm_footprint_t rec, uint64_t now, int who,
                                   unsigned x, unsigned y, unsigned w,
                                   unsigned h);

int ezm_footprint_toggle_heat_mode (ezm_footprint_t rec);

#endif
