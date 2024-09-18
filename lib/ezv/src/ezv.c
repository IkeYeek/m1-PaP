

#include <cglm/cglm.h>
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "ezv_ctx.h"
#include "ezv_hud.h"
#include "ezv_sdl_gl.h"
#include "ezv_shader.h"
#include "ezv_virtual.h"

#include "ezv_img2d_object.h"
#include "ezv_mesh3d_object.h"

#include "mesh3d_renderer.h"

void ezv_init (const char *prefix)
{
  ezv_prefix = prefix;

  if (!SDL_WasInit (SDL_INIT_VIDEO)) {
    int r = SDL_Init (SDL_INIT_VIDEO);
    if (r < 0)
      exit_with_error ("Video initialization failed: %s", SDL_GetError ());
  }
}

void ezv_load_opengl (void)
{
#ifdef USE_GLAD
  static int done = 0;

  if (!done) {
    if (!gladLoadGLLoader ((GLADloadproc)SDL_GL_GetProcAddress))
      exit_with_error ("Failed to initialize GLAD");
    fprintf (stderr, "OpenGL dynamically loaded, version %s\n",
             glGetString (GL_VERSION));
    done = 1;
  }
#endif
}

ezv_ctx_t ezv_ctx_create (ezv_ctx_type_t ctx_type, const char *win_title, int x,
                          int y, int w, int h, int flags)
{
  SDL_Renderer *ren = NULL;
  ezv_ctx_t ctx     = NULL;

  ctx = (ezv_ctx_t)malloc (sizeof (struct ezv_ctx_s));

  ctx->type       = ctx_type;
  ctx->winw       = w;
  ctx->winh       = h;
  ctx->cpu_colors = NULL;
  ezv_palette_init (&ctx->cpu_palette);
  ezv_palette_init (&ctx->data_palette);
  ctx->hud_ctx          = NULL;
  ctx->picking_enabled  = (flags & EZV_ENABLE_PICKING) ? 1 : 0;
  ctx->hud_enabled      = (flags & EZV_ENABLE_HUD) ? 1 : 0;
  ctx->clipping_enabled = ((flags & EZV_ENABLE_CLIPPING) && (ctx_type != EZV_CTX_TYPE_IMG2D)) ? 1 : 0;
  ctx->clipping_active  = 0;
  ctx->object           = NULL;
  ctx->class            = NULL;

  SDL_GL_SetAttribute (SDL_GL_CONTEXT_PROFILE_MASK,
                       SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef __APPLE__
  SDL_GL_SetAttribute ((SDL_GLattr)SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1);
#endif

  ctx->win = SDL_CreateWindow (win_title, x, y, w, h,
                               SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
  SDL_RaiseWindow (ctx->win);

  ctx->windowID = SDL_GetWindowID (ctx->win);

  unsigned drivers     = SDL_GetNumRenderDrivers ();
  int choosen_renderer = -1;

  for (int d = 0; d < drivers; d++) {
    SDL_RendererInfo info;
    SDL_GetRenderDriverInfo (d, &info);
    if (!strcmp (info.name, "opengl"))
      choosen_renderer = d;
  }

  // Initialisation du moteur de rendu
  ren =
      SDL_CreateRenderer (ctx->win, choosen_renderer, SDL_RENDERER_ACCELERATED);
  if (ren == NULL)
    exit_with_error ("SDL_CreateRenderer failed (%s)", SDL_GetError ());

  SDL_RendererInfo info;
  SDL_GetRendererInfo (ren, &info);
  // fprintf (stderr, "Main window renderer used: [%s]\n", info.name);

  // Just in case it is not already loaded
  ezv_load_opengl ();

  ctx->glcontext = SDL_GL_CreateContext (ctx->win);
  if (ctx->glcontext == NULL)
    exit_with_error ("SDL_GL_CreateContext failed (%s)", SDL_GetError ());

  if (flags & EZV_ENABLE_VSYNC) {
    int r = SDL_GL_SetSwapInterval (1);
    if (r != 0)
      fprintf (stderr, "WARNING: SDL_GL_SetSwapInterval is not supported by your OpenGL driver (%s)",
                       (char *)glGetString (GL_RENDERER));
  }

  switch (ctx_type) {
  case EZV_CTX_TYPE_IMG2D: {
    // Initialize img2d specific data
    ezv_img2d_object_init (ctx);
    break;
  }
  case EZV_CTX_TYPE_MESH3D: {
    // Initialize mesh3d specific data
    ezv_mesh3d_object_init (ctx);
    break;
  }
  default:
    exit_with_error ("ctx_type %d not supported", ctx_type);
  }

  // Initialize hud engine
  if (ctx->hud_enabled)
    ezv_hud_init (ctx);

  return ctx;
}

void ezv_ctx_raise (ezv_ctx_t ctx)
{
  SDL_RaiseWindow (ezv_sdl_window (ctx));
}

void ezv_switch_to_context (ezv_ctx_t ctx)
{
  SDL_GL_MakeCurrent (ctx->win, ctx->glcontext);
}

void ezv_ctx_destroy (ezv_ctx_t ctx)
{
  if (ctx->object != NULL)
    free (ctx->object);
  if (ctx->hud_ctx != NULL)
    free (ctx->hud_ctx);
  free (ctx);
}

void ezv_reset_cpu_colors (ezv_ctx_t ctx)
{
  if (!ezv_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not initialized");

  memset (ctx->cpu_colors, 0,
          ezv_get_color_data_size (ctx) * sizeof (unsigned));
}

static void alloc_cpu_colors (ezv_ctx_t ctx)
{
  ctx->cpu_colors = malloc (ezv_get_color_data_size (ctx) * sizeof (unsigned));
  ezv_reset_cpu_colors (ctx);
}

void ezv_set_cpu_color_1D (ezv_ctx_t ctx, unsigned offset, unsigned size,
                           uint32_t color)
{
  if (!ezv_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not initialized");

  if (ezv_c2a (color) == 255) {
    // Don't waste time with unnecessary blend computations
    for (unsigned c = offset; c < offset + size; c++)
      ctx->cpu_colors[c] = color;
  } else {
    vec4 src = {(float)ezv_c2r (color), (float)ezv_c2g (color),
                (float)ezv_c2b (color), 255.0};
    vec4 tmp;
    float srcA = (float)ezv_c2a (color) * (1.0f / 255.0f); // srcA in [0.0..1.0]
    for (unsigned c = offset; c < offset + size; c++) {
      // Blending formula:
      //   dstA = 1 * srcA + dstA * (1 - srcA)
      //   dstRGB = srcRGB * srcA + dstRGB * (1 - srcA)
      uint32_t d = ctx->cpu_colors[c];
      vec4 dst   = {(float)ezv_c2r (d), (float)ezv_c2g (d), (float)ezv_c2b (d),
                    (float)ezv_c2a (d)};
      glm_vec4_mix (dst, src, srcA, tmp);
      ctx->cpu_colors[c] = ezv_rgba ((uint8_t)tmp[0], (uint8_t)tmp[1],
                                     (uint8_t)tmp[2], (u_int8_t)tmp[3]);
    }
  }
}

void ezv_set_cpu_color_2D (ezv_ctx_t ctx, unsigned x, unsigned width,
                           unsigned y, unsigned height, uint32_t color)
{
  if (!ezv_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not initialized");

  unsigned pitch = ezv_get_linepitch (ctx);
  if (ezv_c2a (color) == 255) {
    // Don't waste time with unnecessary blend computations
    for (unsigned i = y; i < y + height; i++)
      for (unsigned j = x; j < x + width; j++)
        ctx->cpu_colors[i * pitch + j] = color;
  } else {
    vec4 src = {(float)ezv_c2r (color), (float)ezv_c2g (color),
                (float)ezv_c2b (color), 255.0};
    vec4 tmp;
    float srcA = (float)ezv_c2a (color) * (1.0f / 255.0f); // srcA in [0.0..1.0]
    for (unsigned i = y; i < y + height; i++)
      for (unsigned j = x; j < x + width; j++) {
        // Blending formula:
        //   dstA = 1 * srcA + dstA * (1 - srcA)
        //   dstRGB = srcRGB * srcA + dstRGB * (1 - srcA)
        uint32_t d = ctx->cpu_colors[i * pitch + j];
        vec4 dst = {(float)ezv_c2r (d), (float)ezv_c2g (d), (float)ezv_c2b (d),
                    (float)ezv_c2a (d)};
        glm_vec4_mix (dst, src, srcA, tmp);
        ctx->cpu_colors[i * pitch + j] =
            ezv_rgba ((uint8_t)tmp[0], (uint8_t)tmp[1], (uint8_t)tmp[2],
                      (u_int8_t)tmp[3]);
      }
  }
}

void ezv_use_cpu_colors (ezv_ctx_t ctx)
{
  ezv_palette_set_RGBA_passthrough (&ctx->cpu_palette);
  alloc_cpu_colors (ctx);
  ezv_activate_rgba_palette (ctx);
}

void ezv_use_data_colors_predefined (ezv_ctx_t ctx, ezv_palette_name_t name)
{
  ezv_palette_set_predefined (&ctx->data_palette, name);
  ezv_activate_data_palette (ctx);
}

void ezv_use_data_colors (ezv_ctx_t ctx, float *data, unsigned size)
{
  if (size < 2)
    exit_with_error (
        "Size (%d) too small: at least two colors must be provided", size);

  ezv_palette_set_raw (&ctx->data_palette, data, size);
  ezv_activate_data_palette (ctx);
}

void ezv_toggle_clipping (ezv_ctx_t ctx[], unsigned nb_ctx)
{
  for (int i = 0; i < nb_ctx; i++)
    if (ctx[i]->clipping_enabled)
      ctx[i]->clipping_active ^= 1;

  ezv_move_zplane (ctx, nb_ctx, 0.0f);
}

char *ezv_ctx_type (ezv_ctx_t ctx)
{
  switch (ctx->type) {
    case EZV_CTX_TYPE_IMG2D:
      return "IMG2D";
    case EZV_CTX_TYPE_MESH3D:
      return "MESH3D";
    default:
      return "UNDEFINED";
  }
}
