#ifndef EZV_CTX_H
#define EZV_CTX_H

#include "ezv_sdl_gl.h"
#include "ezv.h"
#include "ezv_virtual.h"


struct render_ctx_s;
struct hud_ctx_s;

typedef struct ezv_ctx_s
{
  ezv_ctx_type_t type;
  SDL_Window *win;
  int windowID;
  int winw, winh;
  int picking_enabled, hud_enabled, clipping_enabled;
  int clipping_active;
  SDL_GLContext glcontext;
  ezv_palette_t cpu_palette, data_palette;
  uint32_t *cpu_colors;
  struct hud_ctx_s *hud_ctx;
  ezv_class_t *class;
  void *object; // MESH3D, IMG2D, etc.
} *ezv_ctx_t;

enum
{
  BINDING_POINT_COLORBUF,
  BINDING_POINT_MATRICES,
  BINDING_POINT_CUSTOM_COLORS,
  BINDING_POINT_CLIPPING,
  BINDING_POINT_HUDINFO
};


#endif
