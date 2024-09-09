#ifndef IMG2D_RENDERER_H
#define IMG2D_RENDERER_H

#include "ezv.h"

void img2d_renderer_init (ezv_ctx_t ctx);
void img2d_renderer_set_img (ezv_ctx_t ctx);
void img2d_renderer_use_cpu_palette (ezv_ctx_t ctx);
void img2d_renderer_use_data_palette (ezv_ctx_t ctx);
void img2d_renderer_do_picking (ezv_ctx_t ctx, int mousex, int mousey, int *x, int *y);

void img2d_set_data_colors (ezv_ctx_t ctx, void *values);

void img2d_renderer_mvp_update (ezv_ctx_t ctx[], unsigned nb_ctx, float dx,
                                 float dy, float dz);
void img2d_renderer_zplane_update (ezv_ctx_t ctx[], unsigned nb_ctx, float dz);
void img2d_render (ezv_ctx_t ctx);
void img2d_reset_view (ezv_ctx_t ctx[], unsigned nb_ctx);
void img2d_get_shareable_buffer_ids (ezv_ctx_t ctx, int buffer_ids[]);
void img2d_set_data_brightness (ezv_ctx_t ctx, float brightness);

int img2d_renderer_zoom (ezv_ctx_t ctx[], unsigned nb_ctx, unsigned shift_mod,
                          unsigned in);
int img2d_renderer_motion (ezv_ctx_t ctx[], unsigned nb_ctx, int dx, int dy,
                            unsigned wheel);

#endif
