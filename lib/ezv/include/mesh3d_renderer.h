#ifndef MESH3D_RENDERER_H
#define MESH3D_RENDERER_H

#include "ezv.h"

void mesh3d_renderer_init (ezv_ctx_t ctx);
void mesh3d_renderer_set_mesh (ezv_ctx_t ctx);
void mesh3d_renderer_use_cpu_palette (ezv_ctx_t ctx);
void mesh3d_renderer_use_data_palette (ezv_ctx_t ctx);
int mesh3d_renderer_do_picking (ezv_ctx_t ctx, int x, int y);

void mesh3d_set_data_colors (ezv_ctx_t ctx, void *values);

void mesh3d_renderer_mvp_update (ezv_ctx_t ctx[], unsigned nb_ctx, float dx,
                                 float dy, float dz);
void mesh3d_renderer_zplane_update (ezv_ctx_t ctx[], unsigned nb_ctx, float dz);
void mesh3d_render (ezv_ctx_t ctx);
void mesh3d_reset_view (ezv_ctx_t ctx[], unsigned nb_ctx);
void mesh3d_switch_data_color_buffer (ezv_ctx_t ctx);
void mesh3d_get_shareable_buffer_ids (ezv_ctx_t ctx, int buffer_ids[]);
void mesh3d_set_data_brightness (ezv_ctx_t ctx, float brightness);

int mesh3d_renderer_zoom (ezv_ctx_t ctx[], unsigned nb_ctx, unsigned shift_mod,
                          unsigned in);
int mesh3d_renderer_motion (ezv_ctx_t ctx[], unsigned nb_ctx, int dx, int dy,
                            unsigned wheel);

#endif
