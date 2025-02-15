#ifndef EZV_MESH3D_H
#define EZV_MESH3D_H

#include "mesh3d_obj.h"

struct ezv_ctx_s;
typedef struct ezv_ctx_s *ezv_ctx_t;

void ezv_mesh3d_set_mesh (ezv_ctx_t ctx, mesh3d_obj_t *mesh);
void ezv_mesh3d_refresh_mesh (ezv_ctx_t ctx[], unsigned nb_ctx);

#endif
