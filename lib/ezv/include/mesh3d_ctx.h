#ifndef MESH3D_CTX_H
#define MESH3D_CTX_H

#include "mesh3d_sdl_gl.h"
#include "mesh3d_palette.h"
#include "mesh3d_obj.h"
#include "mesh3d_hud.h"
#include "mesh3d.h"

struct render_ctx_s;

typedef struct mesh3d_ctx_s
{
  SDL_Window *win;
  int windowID;
  int winw, winh;
  int picking_enabled, hud_enabled, clipping_enabled;
  int clipping_active;
  SDL_GLContext glcontext;
  mesh3d_obj_t *mesh;
  mesh3d_palette_t cpu_palette, data_palette;
  GLuint *cpu_colors;
  struct render_ctx_s *render_ctx;
  hud_t hud[MAX_HUDS];
} *mesh3d_ctx_t;

#endif
