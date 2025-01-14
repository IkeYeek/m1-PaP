#ifndef EZM_PERFMETER_H
#define EZM_PERFMETER_H

#include "ezv.h"

#include <stdint.h>

struct ezm_perfm_struct;
typedef struct ezm_perfm_struct *ezm_perfmeter_t;

ezm_perfmeter_t ezm_perfmeter_create (unsigned nb_cpus, unsigned nb_gpus,
                                      ezv_ctx_t ctx);

void ezm_perfmeter_enable (ezm_perfmeter_t rec);
void ezm_perfmeter_disable (ezm_perfmeter_t rec);

void ezm_perfmeter_destroy (ezm_perfmeter_t rec);

void ezm_perfmeter_it_start (ezm_perfmeter_t rec, uint64_t now);
void ezm_perfmeter_it_end (ezm_perfmeter_t rec, uint64_t now);
void ezm_perfmeter_start_work (ezm_perfmeter_t rec, uint64_t now, int who);
uint64_t ezm_perfmeter_finish_work (ezm_perfmeter_t rec, uint64_t now, int who);
void ezm_perfmeter_substract_overhead (ezm_perfmeter_t rec, uint64_t duration,
                                     int who);

#endif
