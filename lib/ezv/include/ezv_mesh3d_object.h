#ifndef EVZ_MESH3D_OBJECT_H
#define EVZ_MESH3D_OBJECT_H

#include "ezv_mesh3d.h"
#include "ezv_ctx.h"

struct render_ctx_s;

typedef struct
{
  mesh3d_obj_t *mesh;
  struct render_ctx_s *render_ctx;
} ezv_mesh3d_object_t;

void ezv_mesh3d_object_init (ezv_ctx_t ctx);
void ezv_mesh3d_set_renderer (ezv_ctx_t ctx, struct render_ctx_s *ren);

static inline mesh3d_obj_t *ezv_mesh3d_mesh (ezv_ctx_t ctx)
{
  // TODO: check that ctx type is MESH3D
  return ((ezv_mesh3d_object_t *)(ctx->object))->mesh;
}

static inline struct render_ctx_s *ezv_mesh3d_renderer (ezv_ctx_t ctx)
{
  // TODO: check that ctx type is MESH3D
  return ((ezv_mesh3d_object_t *)(ctx->object))->render_ctx;
}


#endif
