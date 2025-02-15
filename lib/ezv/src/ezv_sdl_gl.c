#include "ezv_sdl_gl.h"
#include "ezv_ctx.h"

SDL_Window *ezv_sdl_window (ezv_ctx_t ctx)
{
  return ctx->win;
}

SDL_GLContext ezv_glcontext (ezv_ctx_t ctx)
{
  return ctx->glcontext;
}
