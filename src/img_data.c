#include <cglm/cglm.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "constants.h"
#include "cppdefs.h"
#include "debug.h"
#include "error.h"
#include "ezp_alloc.h"
#include "ezp_ctx.h"
#include "ezv_event.h"
#include "global.h"
#include "gpu.h"
#include "hooks.h"
#include "img_data.h"

#define THUMBNAILS_SIZE 512

uint32_t *RESTRICT image     = NULL;
uint32_t *RESTRICT alt_image = NULL;

unsigned DIM = 0;

unsigned TILE_W     = 0;
unsigned TILE_H     = 0;
unsigned NB_TILES_X = 0;
unsigned NB_TILES_Y = 0;

static ezv_palette_name_t the_data_palette = EZV_PALETTE_UNDEFINED;

img2d_obj_t easypap_img_desc;

static int picked_x  = -1;
static int picked_y  = -1;
static int coord_hud = -1;
static int tile_hud  = -1;
static int val_hud   = -1;

static unsigned default_tile_size (void)
{
  return gpu_used ? DEFAULT_GPU_TILE_SIZE : DEFAULT_CPU_TILE_SIZE;
}

static void check_tile_size (void)
{
  if (TILE_W == 0) {
    if (NB_TILES_X == 0) {
      TILE_W     = default_tile_size ();
      NB_TILES_X = DIM / TILE_W;
    } else {
      TILE_W = DIM / NB_TILES_X;
    }
  } else if (NB_TILES_X == 0) {
    NB_TILES_X = DIM / TILE_W;
  } else if (NB_TILES_X * TILE_W != DIM) {
    exit_with_error (
        "Inconsistency detected: NB_TILES_X (%d) x TILE_W (%d) != DIM (%d).",
        NB_TILES_X, TILE_W, DIM);
  }

  if (DIM % TILE_W)
    exit_with_error ("DIM (%d) is not a multiple of TILE_W (%d)!", DIM, TILE_W);

  if (TILE_H == 0) {
    if (NB_TILES_Y == 0) {
      TILE_H     = default_tile_size ();
      NB_TILES_Y = DIM / TILE_H;
    } else {
      TILE_H = DIM / NB_TILES_Y;
    }
  } else if (NB_TILES_Y == 0) {
    NB_TILES_Y = DIM / TILE_H;
  } else if (NB_TILES_Y * TILE_H != DIM) {
    exit_with_error (
        "Inconsistency detected: NB_TILES_Y (%d) x TILE_H (%d) != DIM (%d).",
        NB_TILES_Y, TILE_H, DIM);
  }

  if (DIM % TILE_H)
    exit_with_error ("DIM (%d) is not a multiple of TILE_H (%d)!", DIM, TILE_H);
}

void img_data_init (void)
{
  if (easypap_image_file != NULL) {
    img2d_obj_init_from_file (&easypap_img_desc, easypap_image_file);
    if (easypap_img_desc.width != easypap_img_desc.height)
      exit_with_error ("EasyPAP only supports square images so far (width: %d "
                       "!= height: %d)",
                       easypap_img_desc.width, easypap_img_desc.height);
    DIM = easypap_img_desc.height;
  } else {
    if (!DIM)
      DIM = DEFAULT_DIM;

    img2d_obj_init (&easypap_img_desc, DIM, DIM);
  }

  check_tile_size ();

  PRINT_DEBUG ('i', "Init phase 0 (IMG2D mode) : DIM = %d\n", DIM);
}

void img_data_alloc (void)
{
  const unsigned size = DIM * DIM * sizeof (uint32_t);

  image     = ezp_alloc (size);
  alt_image = ezp_alloc (size);

  PRINT_DEBUG ('i', "Init phase 4: images allocated\n");
}

void img_data_imgload (void)
{
  if (easypap_image_file != NULL)
    img2d_obj_load (&easypap_img_desc, easypap_image_file, image);
}

void img_data_free (void)
{
  const unsigned size = DIM * DIM * sizeof (uint32_t);

  ezp_free (image, size);
  image = NULL;

  ezp_free (alt_image, size);
  alt_image = NULL;
}

void img_data_replicate (void)
{
  memcpy (alt_image, image, DIM * DIM * sizeof (uint32_t));
}

void img_data_set_default_palette_if_none_defined (void)
{
  if (do_display) {
    if (nb_ctx < 1)
      exit_with_error ("No img2d ctx created yet");

    if (the_data_palette == EZV_PALETTE_UNDEFINED) {
      the_data_palette = EZV_PALETTE_RGBA_PASSTHROUGH;
      ezv_use_data_colors_predefined (ctx[0], the_data_palette);
    }
  }
}

void img_data_init_huds (int show)
{
  ezp_ctx_ithud_init (show);

  if (picking_enabled) {
    coord_hud = ezv_hud_alloc (ctx[0]);
    tile_hud  = ezv_hud_alloc (ctx[0]);
    if (the_2d_debug == NULL) {
      val_hud = ezv_hud_alloc (ctx[0]);
      ezv_hud_on (ctx[0], val_hud);
    }
  }
}

void img_data_refresh (unsigned iter)
{
  ezp_ctx_ithud_set (iter);

  if (picking_enabled) {
    if (the_2d_debug != NULL)
      the_2d_debug (picked_x, picked_y);
    else {
      if (picked_x == -1 || picked_y == -1)
        ezv_hud_set (ctx[0], val_hud, NULL);
      else {
        uint32_t v = cur_img (picked_y, picked_x);
        ezv_hud_set (ctx[0], val_hud, "RGBA: %02X %02X %02X %02X", ezv_c2r (v),
                     ezv_c2g (v), ezv_c2b (v), ezv_c2a (v));
      }
    }
  }

  // If computations were performed on CPU (that is, in the 'image' array), copy
  // data into texture buffer. Otherwise (GPU), data are already in place
  if (!gpu_used || !easypap_gl_buffer_sharing)
    ezv_set_data_colors (ctx[0], image);
  else
    gpu_update_texture ();

  ezv_render (ctx, nb_ctx);
}

void img_data_dump_to_file (char *filename)
{
  img2d_obj_store (&easypap_img_desc, filename, image);
}

void img_data_save_thumbnail (unsigned iteration)
{
  char filename[1024];

  sprintf (filename, "data/traces/thumb_%04d.png", iteration);

  if (easypap_img_desc.width > THUMBNAILS_SIZE) {
    img2d_obj_store_resized (&easypap_img_desc, filename, image,
                             THUMBNAILS_SIZE, THUMBNAILS_SIZE);
  } else {
    // Image is small enough, so store as is
    img2d_obj_store (&easypap_img_desc, filename, image);
  }
}

void img_data_do_pick (void)
{
  int px, py;

  ezv_perform_2D_picking (ctx, 1, &px, &py);

  if (px != picked_x || py != picked_y) { // focus has changed
    picked_x = px;
    picked_y = py;

    ezv_reset_cpu_colors (ctx[0]);

    if (px != -1 && py != -1) {
      ezv_hud_on (ctx[0], coord_hud);
      ezv_hud_set (ctx[0], coord_hud, "xy: (%d, %d)", px, py);

      int tilex = px / TILE_W;
      int tiley = py / TILE_H;

      ezv_hud_on (ctx[0], tile_hud);
      ezv_hud_set (ctx[0], tile_hud, "Tilexy: (%d, %d)", tilex, tiley);

      if (the_2d_overlay != NULL)
        the_2d_overlay (px, py);
      else {
        // highlight tile
        ezv_set_cpu_color_2D (ctx[0], tilex * TILE_W, TILE_W, tiley * TILE_H,
                              TILE_H, ezv_rgba (0xFF, 0xFF, 0xFF, 0xA0));
        // highlight pixel
        ezv_set_cpu_color_2D (ctx[0], px, 1, py, 1,
                              ezv_rgba (0xFF, 0x00, 0x00, 0xC0));
      }
    } else {
      ezv_hud_off (ctx[0], coord_hud);
      ezv_hud_off (ctx[0], tile_hud);
    }
  }
}

// Color utilities

const int heat_size       = 5;
static vec4 heat_colors[] = {{0.0f, 0.0f, 1.0f, 1.0f},  // blue
                             {0.0f, 1.0f, 1.0f, 1.0f},  // cyan
                             {0.0f, 1.0f, 0.0f, 1.0f},  // green
                             {1.0f, 1.0f, 0.0f, 1.0f},  // yellow
                             {1.0f, 0.0f, 0.0f, 1.0f}}; // red

static unsigned val_to_rgba (float h, vec4 colors[], int size)
{
  float scale = h * (float)(size - 1);
  int ind     = scale;
  float frac;
  
  if (ind < size - 1)
    frac = scale - ind;
  else {
    frac = 1.0;
    ind--;
  }

  vec4 theColor;
  glm_vec4_mix (colors[ind], colors[ind + 1], frac, theColor);
  glm_vec4_scale (theColor, 255.0, theColor);

  return ezv_rgba (theColor[0], theColor[1], theColor[2], theColor[3]);
}

unsigned heat_to_rgb (float v) // 0.0 = cold, 1.0 = hot
{
  return val_to_rgba (v, heat_colors, heat_size);
}

const int gauss_size       = 7;
static vec4 gauss_colors[] = {{0.0, 0.0, 1.0, 1.0},       // blue
                              {0.1667, 1.0, 0.8333, 1.0}, //
                              {0.3333, 0.0, 0.6666, 1.0}, //
                              {0.5, 0.0, 0.5, 1.0},       //
                              {0.6666, 1.0, 0.3333, 1.0}, //
                              {0.8333, 1.0, 0.1667, 1.0}, //
                              {1.0, 0.0, 0.0, 1.0}};      // red

unsigned heat_to_3gauss_rgb (double v) // v is between 0.0 and 1.0
{
  return val_to_rgba (v, gauss_colors, gauss_size);
}
