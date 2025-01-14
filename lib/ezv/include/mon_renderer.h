#ifndef MON_RENDERER_H
#define MON_RENDERER_H

#include "ezv.h"

void mon_renderer_init (ezv_ctx_t ctx);
void mon_renderer_set_mon (ezv_ctx_t ctx);
void mon_renderer_use_cpu_palette (ezv_ctx_t ctx);
void mon_renderer_use_data_palette (ezv_ctx_t ctx);

void mon_set_data_colors (ezv_ctx_t ctx, void *values);

void mon_render (ezv_ctx_t ctx);


#endif
