#ifndef GL_RENDERER_IS_DEF
#define GL_RENDERER_IS_DEF

#include "mesh3d.h"

void mesh3d_renderer_init (mesh3d_ctx_t ctx);
void mesh3d_renderer_set_mesh (mesh3d_ctx_t ctx);
void mesh3d_renderer_use_cpu_palette (mesh3d_ctx_t ctx);
void mesh3d_renderer_use_data_palette (mesh3d_ctx_t ctx);
int mesh3d_renderer_do_picking (mesh3d_ctx_t ctx, int x, int y);

void mesh3d_renderer_mvp_update (mesh3d_ctx_t ctx[], unsigned nb_ctx,
                                 float dx, float dy, float dz);
void mesh3d_renderer_zplane_update (mesh3d_ctx_t ctx[], unsigned nb_ctx,
                                    float dz);

#endif
