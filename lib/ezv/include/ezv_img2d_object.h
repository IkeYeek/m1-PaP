#ifndef EVZ_IMG2D_OBJECT_H
#define EVZ_IMG2D_OBJECT_H

#include "ezv_img2d.h"
#include "ezv_ctx.h"

struct img2d_render_ctx_s;

typedef struct
{
  img2d_obj_t *img;
  struct img2d_render_ctx_s *render_ctx;
} ezv_img2d_object_t;

void ezv_img2d_object_init (ezv_ctx_t ctx);
void ezv_img2d_set_renderer (ezv_ctx_t ctx, struct img2d_render_ctx_s *ren);

static inline img2d_obj_t *ezv_img2d_img (ezv_ctx_t ctx)
{
  // TODO: check that ctx type is IMG2D
  return ((ezv_img2d_object_t *)(ctx->object))->img;
}

static inline struct img2d_render_ctx_s *ezv_img2d_renderer (ezv_ctx_t ctx)
{
  // TODO: check that ctx type is IMG2D
  return ((ezv_img2d_object_t *)(ctx->object))->render_ctx;
}


#endif
