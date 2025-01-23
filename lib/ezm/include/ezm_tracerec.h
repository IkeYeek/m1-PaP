#ifndef EZM_TRACEREC_H
#define EZM_TRACEREC_H

#include "ezm_trace_codes.h"
#include "ezm_types.h"
#include "ezv.h"

#include <stdint.h>

struct ezm_trrec_struct;
typedef struct ezm_trrec_struct *ezm_tracerec_t;

ezm_tracerec_t ezm_tracerec_create (unsigned nb_cpus, unsigned nb_gpus,
                                    char *file, char *label);
void ezm_tracerec_destroy (ezm_tracerec_t rec);

void ezm_tracerec_it_start (ezm_tracerec_t rec);
void ezm_tracerec_it_end (ezm_tracerec_t rec);

void ezm_tracerec_start_work (ezm_tracerec_t rec, int cpu);

void ezm_tracerec_1D (ezm_tracerec_t rec, int cpu, unsigned patch,
                      unsigned count);
void ezm_tracerec_1D_task (ezm_tracerec_t rec, int cpu, unsigned patch,
                           unsigned count, int task_id);
void ezm_tracerec_1D_ext (ezm_tracerec_t rec, uint64_t start_time,
                          uint64_t end_time, int cpu, unsigned patch,
                          unsigned count, task_type_t task_type, int task_id);

void ezm_tracerec_2D (ezm_tracerec_t rec, int cpu, unsigned x, unsigned y,
                      unsigned w, unsigned h);
void ezm_tracerec_2D_task (ezm_tracerec_t rec, int cpu, unsigned x, unsigned y,
                           unsigned w, unsigned h, int task_id);
void ezm_tracerec_2D_ext (ezm_tracerec_t rec, uint64_t start_time,
                          uint64_t end_time, int cpu, unsigned x, unsigned y,
                          unsigned w, unsigned h, task_type_t task_type,
                          int task_id);

void ezm_tracerec_enable (ezm_tracerec_t rec, unsigned iter);
void ezm_tracerec_disable (ezm_tracerec_t rec);

void ezm_tracerec_store_mesh3d_filename (ezm_tracerec_t rec,
                                         const char *filename);
void ezm_tracerec_store_img2d_dim (ezm_tracerec_t rec, unsigned width,
                                   unsigned height);
void ezm_tracerec_store_data_palette (ezm_tracerec_t rec,
                                      ezv_palette_name_t pal);
void ezm_tracerec_declare_task_ids (ezm_tracerec_t rec, char *task_ids[]);

#endif
