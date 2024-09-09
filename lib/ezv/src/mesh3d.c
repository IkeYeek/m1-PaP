

#include <cglm/cglm.h>
#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "mesh3d_ctx.h"
#include "mesh3d_renderer.h"
#include "mesh3d_sdl_gl.h"
#include "mesh3d_shader.h"

static int mouse[2]     = {-1, -1};
static int active_winID = -1;

void mesh3d_init (const char *prefix)
{
  mesh3d_prefix = prefix;

  if (!SDL_WasInit (SDL_INIT_VIDEO)) {
    int r = SDL_Init (SDL_INIT_VIDEO);
    if (r < 0)
      exit_with_error ("Video initialization failed: %s", SDL_GetError ());
  }
}

void mesh3d_load_opengl (void)
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

mesh3d_ctx_t mesh3d_ctx_create (const char *win_title, int x, int y, int w,
                                int h, int flags)
{
  SDL_Renderer *ren = NULL;
  mesh3d_ctx_t ctx  = NULL;

  ctx = (mesh3d_ctx_t)malloc (sizeof (struct mesh3d_ctx_s));

  ctx->mesh       = NULL;
  ctx->winw       = w;
  ctx->winh       = h;
  ctx->cpu_colors = NULL;
  mesh3d_palette_init (&ctx->cpu_palette);
  mesh3d_palette_init (&ctx->data_palette);
  ctx->picking_enabled = (flags & MESH3D_ENABLE_PICKING) ? 1 : 0;
  ctx->hud_enabled     = (flags & MESH3D_ENABLE_HUD) ? 1 : 0;
  mesh3d_hud_init (ctx);
  ctx->clipping_enabled = (flags & MESH3D_ENABLE_CLIPPING) ? 1 : 0;
  ctx->clipping_active  = 0;

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
  mesh3d_load_opengl ();

  ctx->glcontext = SDL_GL_CreateContext (ctx->win);
  if (ctx->glcontext == NULL)
    exit_with_error ("SDL_GL_CreateContext failed (%s)", SDL_GetError ());

  if (flags & MESH3D_ENABLE_VSYNC) {
    int r = SDL_GL_SetSwapInterval (1);
    if (r != 0)
      exit_with_error ("SDL_GL_SetSwapInterval not supported (%s)",
                       SDL_GetError ());
  }

  mesh3d_renderer_init (ctx);

  return ctx;
}

SDL_Window *mesh3d_sdl_window (mesh3d_ctx_t ctx)
{
  return ctx->win;
}

SDL_GLContext mesh3d_glcontext (mesh3d_ctx_t ctx)
{
  return ctx->glcontext;
}

void mesh3d_ctx_destroy (mesh3d_ctx_t ctx)
{
  free (ctx->render_ctx);
  free (ctx);
}

void mesh3d_set_mesh (mesh3d_ctx_t ctx, mesh3d_obj_t *mesh)
{
  ctx->mesh = mesh;
  // tell renderer that mesh is known
  mesh3d_renderer_set_mesh (ctx);
}

void mesh3d_reset_cpu_colors (mesh3d_ctx_t ctx)
{
  if (!mesh3d_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not initialized");

  memset (ctx->cpu_colors, 0, ctx->mesh->nb_cells * sizeof (unsigned));
}

static void mesh3d_init_cpu_colors (mesh3d_ctx_t ctx)
{
  ctx->cpu_colors = malloc (ctx->mesh->nb_cells * sizeof (unsigned));
  mesh3d_reset_cpu_colors (ctx);
}

void mesh3d_set_cpu_color (mesh3d_ctx_t ctx, unsigned first_cell,
                           unsigned num_cells, unsigned color)
{
  if (!mesh3d_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not initialized");

  if ((color & 0xFF) == 0xFF) {
    // Don't waste time with unnecessary blend computations
    for (unsigned c = first_cell; c < first_cell + num_cells; c++)
      ctx->cpu_colors[c] = color;
  } else
    for (unsigned c = first_cell; c < first_cell + num_cells; c++) {
      // FIXME: optimize packing/unpacking ops?
      // Blending formula:
      //   dstA = 1 * srcA + dstA * (1 - srcA)
      //   dstRGB = srcRGB * srcA + dstRGB * (1 - srcA)
      unsigned d = ctx->cpu_colors[c];
      vec4 dst   = {(float)(d >> 24), (float)((d >> 16) & 0xFF),
                    (float)((d >> 8) & 0xFF), (float)(d & 0xFF)};
      vec4 src   = {(float)(color >> 24), (float)((color >> 16) & 0xFF),
                    (float)((color >> 8) & 0xFF), 255.0};
      vec4 tmp;
      float srcA =
          (float)(color & 0xFF) * (1.0f / 255.0f); // srcA in [0.0..1.0]
      glm_vec4_mix (dst, src, srcA, tmp);
      ctx->cpu_colors[c] = ((int)tmp[0]) << 24 | ((int)tmp[1]) << 16 |
                           ((int)tmp[2]) << 8 | (int)tmp[3];
    }
}

void mesh3d_use_cpu_colors (mesh3d_ctx_t ctx)
{
  mesh3d_palette_set_RGBA_passthrough (&ctx->cpu_palette);
  mesh3d_init_cpu_colors (ctx);
  mesh3d_renderer_use_cpu_palette (ctx);
}

void mesh3d_configure_data_colors_predefined (mesh3d_ctx_t ctx,
                                              mesh3d_palette_name_t name)
{
  mesh3d_palette_set_predefined (&ctx->data_palette, name);
  mesh3d_renderer_use_data_palette (ctx);
}

void mesh3d_configure_data_colors (mesh3d_ctx_t ctx, float *data, unsigned size)
{
  if (size < 2)
    exit_with_error (
        "Size (%d) too small: at least two colors must be provided", size);

  mesh3d_palette_set_raw (&ctx->data_palette, data, size);
  mesh3d_renderer_use_data_palette (ctx);
}

void mesh3d_mouse_enter (mesh3d_ctx_t ctx[], unsigned nb_ctx, SDL_Event *event)
{
  for (int c = 0; c < nb_ctx; c++)
    if (event->window.windowID == ctx[c]->windowID) {
      active_winID = ctx[c]->windowID;
      return;
    }
}

void mesh3d_mouse_leave (mesh3d_ctx_t ctx[], unsigned nb_ctx, SDL_Event *event)
{
  mouse[0]     = -1;
  mouse[1]     = -1;
  active_winID = -1;
}

void mesh3d_mouse_focus (mesh3d_ctx_t ctx[], unsigned nb_ctx, SDL_Event *event)
{
  mouse[0] = event->motion.x;
  mouse[1] = event->motion.y;
}

int mesh3d_ctx_is_in_focus (mesh3d_ctx_t ctx)
{
  return active_winID == ctx->windowID;
}

static int active_ctx_enables_picking (mesh3d_ctx_t ctx[], unsigned nb_ctx)
{
  for (int c = 0; c < nb_ctx; c++)
    if (active_winID == ctx[c]->windowID)
      return ctx[c]->picking_enabled;

  return 0;
}

int mesh3d_perform_picking (mesh3d_ctx_t ctx[], unsigned nb_ctx)
{
  if (active_winID == -1)
    return -1;

  for (int c = 0; c < nb_ctx; c++)
    if (ctx[c]->picking_enabled && (ctx[c]->windowID == active_winID))
      return mesh3d_renderer_do_picking (ctx[c], mouse[0], mouse[1]);

  return -1;
}

void mesh3d_toggle_clipping (mesh3d_ctx_t ctx[], unsigned nb_ctx)
{
  for (int i = 0; i < nb_ctx; i++)
    if (ctx[i]->clipping_enabled)
      ctx[i]->clipping_active ^= 1;
  mesh3d_renderer_zplane_update (ctx, nb_ctx, 0.0f);
}

static int mouse_click_x = -1;
static int mouse_click_y = -1;
static int button_down   = 0;

// pick is set to 1 if picking should be re-done, 0 otherwise
// refresh is set to 1 if rendering should be performed
void mesh3d_process_event (mesh3d_ctx_t ctx[], unsigned nb_ctx,
                           SDL_Event *event, int *refresh, int *pick)
{
  int do_pick    = 0;
  int do_refresh = 0;

  switch (event->type) {
  case SDL_KEYDOWN:
    switch (event->key.keysym.sym) {
    case SDLK_ESCAPE:
    case SDLK_q:
      // Note: usually handled ahead of this function call
      exit (0);
      break;
    case SDLK_r:
      // Reset view
      mesh3d_reset_view (ctx, nb_ctx);
      do_pick    = active_ctx_enables_picking (ctx, nb_ctx);
      do_refresh = 1;
      break;
    case SDLK_MINUS:
    case SDLK_KP_MINUS:
    case SDLK_m:
    case SDLK_l:
      if (event->key.keysym.mod & KMOD_SHIFT) {
        // Move clipping plane backward
        if (ctx[0]->clipping_active) {
          mesh3d_renderer_zplane_update (ctx, nb_ctx, -0.01f);
          do_pick    = active_ctx_enables_picking (ctx, nb_ctx);
          do_refresh = 1;
        }
      } else {
        // zoom out
        mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, -.02f);
        do_pick    = active_ctx_enables_picking (ctx, nb_ctx);
        do_refresh = 1;
      }
      break;
    case SDLK_PLUS:
    case SDLK_KP_PLUS:
    case SDLK_EQUALS:
    case SDLK_p:
    case SDLK_o:
      if (event->key.keysym.mod & KMOD_SHIFT) {
        // move clipping plane forward
        if (ctx[0]->clipping_active) {
          mesh3d_renderer_zplane_update (ctx, nb_ctx, 0.01f);
          do_pick    = active_ctx_enables_picking (ctx, nb_ctx);
          do_refresh = 1;
        }
      } else {
        // Zoom in
        mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, +.02f);
        do_pick    = active_ctx_enables_picking (ctx, nb_ctx);
        do_refresh = 1;
      }
      break;
    case SDLK_c:
      // Toggle clipping
      mesh3d_toggle_clipping (ctx, nb_ctx);
      do_refresh = 1;
      break;
    }
    break;
  case SDL_QUIT: // normally handled by easypap/easyview
    exit (0);
    break;
  case SDL_MOUSEMOTION:
    mesh3d_mouse_focus (ctx, nb_ctx, event);
    do_pick = active_ctx_enables_picking (ctx, nb_ctx);
    if (button_down) {
      float dx = (float)(event->motion.y - mouse_click_y) * 0.2;
      float dy = (float)(event->motion.x - mouse_click_x) * 0.2;

      mouse_click_x = event->motion.x;
      mouse_click_y = event->motion.y;

      mesh3d_renderer_mvp_update (ctx, nb_ctx, dx, dy, 0.0f);
      do_pick    = active_ctx_enables_picking (ctx, nb_ctx);
      do_refresh = 1;
    }
    break;
  case SDL_MOUSEWHEEL: {
    float dx = 1.5f * event->wheel.y; // not a mistake ;)
    float dy = -1.5f * event->wheel.x;
    mesh3d_renderer_mvp_update (ctx, nb_ctx, dx, dy, 0.0f);
    do_pick    = active_ctx_enables_picking (ctx, nb_ctx);
    do_refresh = 1;
  } break;
  case SDL_MOUSEBUTTONDOWN:
    mouse_click_x = event->button.x;
    mouse_click_y = event->button.y;
    button_down   = 1;
    break;
  case SDL_MOUSEBUTTONUP: {
    button_down = 0;
    break;
  }
  case SDL_WINDOWEVENT:
    switch (event->window.event) {
    case SDL_WINDOWEVENT_ENTER:
      mesh3d_mouse_enter (ctx, nb_ctx, event);
      break;
    case SDL_WINDOWEVENT_LEAVE:
      mesh3d_mouse_leave (ctx, nb_ctx, event);
      do_pick = active_ctx_enables_picking (ctx, nb_ctx);
      break;
    default:;
    }
  }
  if (refresh)
    *refresh = do_refresh;
  if (pick)
    *pick = do_pick;
}
