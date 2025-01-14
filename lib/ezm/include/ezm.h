#ifndef EZM_H
#define EZM_H

#include "ezm_footprint.h"
#include "ezm_perfmeter.h"
#include "ezm_tracerec.h"
#include "ezv.h"

struct ezm_recorder_struct;
typedef struct ezm_recorder_struct *ezm_recorder_t;

#define EZM_NO_DISPLAY 1

void ezm_init (char *prefix, unsigned flags);

ezm_recorder_t ezm_recorder_create (unsigned nb_cpus, unsigned nb_gpus);
void ezm_recorder_destroy (ezm_recorder_t rec);

void ezm_set_cpu_palette (ezm_recorder_t rec, ezv_palette_name_t name,
                          int cyclic_mode);

void ezm_recorder_attach_perfmeter (ezm_recorder_t rec, ezv_ctx_t ctx);
void ezm_recorder_attach_footprint (ezm_recorder_t rec, ezv_ctx_t ctx);
void ezm_recorder_attach_tracerec (ezm_recorder_t rec, char *file, char *label);

ezm_perfmeter_t ezm_recorder_get_perfmeter (ezm_recorder_t rec);
ezm_footprint_t ezm_recorder_get_footprint (ezm_recorder_t rec);
ezm_tracerec_t ezm_recorder_get_tracerec (ezm_recorder_t rec);

void ezm_recorder_enable (ezm_recorder_t rec, unsigned iter);
void ezm_recorder_disable (ezm_recorder_t rec);
int ezm_recorder_is_enabled (ezm_recorder_t rec);

void ezm_start_iteration (ezm_recorder_t rec);
void ezm_end_iteration (ezm_recorder_t rec);

void ezm_start_work (ezm_recorder_t rec, unsigned cpu);

void ezm_end_1D (ezm_recorder_t rec, unsigned cpu, unsigned patch,
                 unsigned count);
void ezm_end_1D_task (ezm_recorder_t rec, unsigned cpu, unsigned patch,
                      unsigned count, int task_id);
void ezm_1D_ext (ezm_recorder_t rec, uint64_t start_time, uint64_t end_time,
                 unsigned cpu, unsigned patch, unsigned count,
                 task_type_t task_type, int task_id);

void ezm_end_2D (ezm_recorder_t rec, unsigned cpu, unsigned x, unsigned y,
                 unsigned w, unsigned h);
void ezm_end_2D_task (ezm_recorder_t rec, unsigned cpu, unsigned x, unsigned y,
                      unsigned w, unsigned h, int task_id);
void ezm_2D_ext (ezm_recorder_t rec, uint64_t start_time, uint64_t end_time,
                 unsigned cpu, unsigned x, unsigned y, unsigned w, unsigned h,
                 task_type_t task_type, int task_id);

// specific functions
void ezm_recorder_toggle_heat_mode (ezm_recorder_t rec);
void ezm_recorder_store_mesh3d_filename (ezm_recorder_t rec,
                                         const char *filename);
void ezm_recorder_store_img2d_dim (ezm_recorder_t rec, unsigned width,
                                   unsigned height);
void ezm_recorder_store_data_palette (ezm_recorder_t rec,
                                      ezv_palette_name_t pal);
void ezm_recorder_declare_task_ids (ezm_recorder_t rec, char *task_ids[]);

// helpers
void ezm_helper_add_perfmeter (ezm_recorder_t rec, ezv_ctx_t ctx[],
                               unsigned *nb_ctx);
void ezm_helper_add_footprint (ezm_recorder_t rec, ezv_ctx_t ctx[],
                               unsigned *nb_ctx);

#endif
