#include <stdio.h>

#include "api_funcs.h"
#include "debug.h"
#include "error.h"
#include "ezp_ctx.h"
#include "ezm.h"
#include "ezv.h"
#include "ezv_sdl_gl.h"
#include "global.h"

static unsigned LARGE_WIN_WIDTH[2];
static unsigned LARGE_WIN_HEIGHT[2];
static unsigned SMALL_WIN_WIDTH[2];
static unsigned SMALL_WIN_HEIGHT[2];

#define MAX_CTX 3

ezv_ctx_t ctx[MAX_CTX] = {NULL, NULL, NULL};
unsigned nb_ctx        = 0;

static int iteration_hud = -1;

void ezp_ctx_init (void)
{
  ezv_init ("lib/ezv");

  LARGE_WIN_WIDTH[EZV_CTX_TYPE_MESH3D]  = 1024;
  LARGE_WIN_HEIGHT[EZV_CTX_TYPE_MESH3D] = 768;
  SMALL_WIN_WIDTH[EZV_CTX_TYPE_MESH3D]  = 512; // 768;
  SMALL_WIN_HEIGHT[EZV_CTX_TYPE_MESH3D] = 384; // 576;

  LARGE_WIN_WIDTH[EZV_CTX_TYPE_IMG2D]  = 1024;
  LARGE_WIN_HEIGHT[EZV_CTX_TYPE_IMG2D] = 1024;
  SMALL_WIN_WIDTH[EZV_CTX_TYPE_IMG2D]  = 512;
  SMALL_WIN_HEIGHT[EZV_CTX_TYPE_IMG2D] = 512;
}

void ezp_ctx_coord_next (ezv_ctx_type_t ctx_type, unsigned ctx_no, int *xwin,
                         int *ywin)
{
  if (ctx_no == 0) {
    if (easypap_mpirun && easypap_mpi_size () > 1 && debug_enabled ('M')) {
      // FIXME: layout computation should take ctx_type into account
      *xwin = (easypap_mpi_rank () % 2) * (LARGE_WIN_WIDTH[ctx_type] / 2 +
                                           SMALL_WIN_WIDTH[ctx_type] / 2 + 352);
      *ywin = (easypap_mpi_rank () / 2) * (LARGE_WIN_HEIGHT[ctx_type] / 2 + 82);
    } else {
      *xwin = 0;
      *ywin = 0;
    }
    return;
  }

  // ctx_no > 0
  SDL_Window *win = ezv_sdl_window (ctx[ctx_no - 1]);
  int x = -1, w = -1;

  SDL_GetWindowPosition (win, &x, ywin);
  SDL_GetWindowSize (win, &w, NULL);

  *xwin = x + w;
}

static void ezp_ctx_dim_next (ezv_ctx_type_t ctx_type, unsigned ctx_no,
                              int *width, int *height)
{
  int w, h;

  if (ctx_no == 0) {
    // Use large dimensions
    w = LARGE_WIN_WIDTH[ctx_type];
    h = LARGE_WIN_HEIGHT[ctx_type];
  } else {
    // Use small dimensions
    w = SMALL_WIN_WIDTH[ctx_type];
    h = SMALL_WIN_HEIGHT[ctx_type];
  }
  if (easypap_mpirun && easypap_mpi_size () > 1 && debug_enabled ('M')) {
    w /= 2;
    h /= 2;
  }
  *width  = w;
  *height = h;
}

int ezp_ctx_create (ezv_ctx_type_t ctx_type)
{
  char title[1024];
  int ctx_no = -1;
  int flags  = EZV_ENABLE_CLIPPING;
  int xpos   = -1;
  int ypos   = -1;
  int width  = -1;
  int height = -1;

  if (!do_display)
    exit_with_error ("ctx cannot be allocated in 'no display' mode");

  if (vsync)
    flags |= EZV_ENABLE_VSYNC;

  ctx_no = nb_ctx++;

  if (ctx_no == 0) {
    const char *subtitle =
        (easypap_mode == EASYPAP_MODE_3D_MESHES) ? "Patching" : "Tiling";

    if (easypap_mpirun) {
      sprintf (title,
               "EasyPAP -- Process: [%d/%d]   Kernel: [%s]   Variant: [%s]   "
               "%s: [%s]",
               easypap_mpi_rank (), easypap_mpi_size (), kernel_name,
               variant_name, subtitle, tile_name);
    } else
      sprintf (title, "EasyPAP -- Kernel: [%s]   Variant: [%s]   %s: [%s]",
               kernel_name, variant_name, subtitle, tile_name);

    flags |= EZV_ENABLE_HUD;
    if (picking_enabled)
      flags |= EZV_ENABLE_PICKING;

  } else {
    strcpy (title, "Tile Mapping");
  }

  // Calculate window layout
  ezp_ctx_coord_next (ctx_type, ctx_no, &xpos, &ypos);
  ezp_ctx_dim_next (ctx_type, ctx_no, &width, &height);

  ctx[ctx_no] =
      ezv_ctx_create (ctx_type, title, xpos, ypos, width, height, flags);

  return ctx_no;
}

void ezp_ctx_ithud_init (int show)
{
  iteration_hud = ezv_hud_alloc (ctx[0]);
  if (show)
    ezv_hud_on (ctx[0], iteration_hud);
}

void ezp_ctx_ithud_toggle (void)
{
  ezv_hud_toggle (ctx[0], iteration_hud);
}

void ezp_ctx_ithud_set (unsigned iter)
{
  ezv_hud_set (ctx[0], iteration_hud, "It: %d", iter);
}
