#ifndef EVZ_MON_OBJECT_H
#define EVZ_MON_OBJECT_H

#include "ezv_mon.h"
#include "ezv_ctx.h"

struct mon_render_ctx_s;

typedef struct
{
  mon_obj_t *mon;
  struct mon_render_ctx_s *render_ctx;
} ezv_mon_object_t;

void ezv_mon_object_init (ezv_ctx_t ctx);
void ezv_mon_set_renderer (ezv_ctx_t ctx, struct mon_render_ctx_s *ren);

static inline mon_obj_t *ezv_mon_mon (ezv_ctx_t ctx)
{
  // TODO: check that ctx type is mon
  return ((ezv_mon_object_t *)(ctx->object))->mon;
}

static inline struct mon_render_ctx_s *ezv_mon_renderer (ezv_ctx_t ctx)
{
  // TODO: check that ctx type is mon
  return ((ezv_mon_object_t *)(ctx->object))->render_ctx;
}


#endif
