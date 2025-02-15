
#include <SDL2/SDL.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "error.h"
#include "ezm.h"
#include "ezv.h"
#include "ezv_event.h"
#include "ezv_sdl_gl.h"

// settings
const unsigned int WIN_WIDTH  = 1024;
const unsigned int WIN_HEIGHT = 1024;

#define MAX_CTX 4

unsigned image_dim     = 1024;
unsigned tile_size     = 32;
static uint32_t *image = NULL;

// ezv/ezm data
static img2d_obj_t img;

// ezv contexts
static ezv_ctx_t ctx[MAX_CTX] = {NULL, NULL, NULL, NULL};
static unsigned nb_ctx        = 0;

// profiling
static ezm_recorder_t recorder = NULL;

static unsigned do_trace = 0;

// Debug HUDs
static int pos_hud = -1;
static int val_hud = -1;

static void create_huds (ezv_ctx_t ctx)
{
  pos_hud = ezv_hud_alloc (ctx);
  ezv_hud_on (ctx, pos_hud);
  val_hud = ezv_hud_alloc (ctx);
  ezv_hud_on (ctx, val_hud);
}

static void do_pick (void)
{
  int x, y;

  ezv_perform_2D_picking (ctx, nb_ctx, &x, &y);

  ezv_hud_set (ctx[0], pos_hud, "xy: (%d, %d)", x, y);

  ezv_reset_cpu_colors (ctx[0]);

  if (x != -1 && y != -1) {
    uint32_t v = image[y * img.width + x];
    ezv_hud_set (ctx[0], val_hud, "RGBA: %02X %02X %02X %02X", ezv_c2r (v),
                 ezv_c2g (v), ezv_c2b (v), ezv_c2a (v));

    int tilex = (x / tile_size) * tile_size;
    int tiley = (y / tile_size) * tile_size;

    // highlight tile
    ezv_set_cpu_color_2D (ctx[0], tilex, tile_size, tiley, tile_size,
                          ezv_rgba (0xFF, 0xFF, 0xFF, 0xA0));
    // highlight pixel
    ezv_set_cpu_color_2D (ctx[0], x, 1, y, 1,
                          ezv_rgba (0xFF, 0x00, 0x00, 0xFF));
  } else
    ezv_hud_set (ctx[0], val_hud, NULL);
}

static void process_events (int blocking)
{
  SDL_Event event;
  int r;

  do {
    r = ezv_get_event (&event, blocking);
    if (r > 0) {
      switch (event.type) {
      case SDL_KEYDOWN:
        switch (event.key.keysym.sym) {
        case SDLK_h:
          ezm_recorder_toggle_heat_mode (recorder);
          break;
        case SDLK_e:
          ezm_recorder_enable (recorder, 0);
          break;
        case SDLK_d:
          ezm_recorder_disable (recorder);
          break;
        default:
          ezv_process_event (ctx, nb_ctx, &event, NULL, NULL);
        }
        break;
      default:
        ezv_process_event (ctx, nb_ctx, &event, NULL, NULL);
      }
    }
  } while (r > 0 && !blocking);
  do_pick ();
}

// mandel computation
static void mandel_init (void);
static uint32_t iteration_to_color (unsigned iter);
static void zoom (void);
static uint32_t compute_one_pixel (int i, int j);
static unsigned nb_openmp_threads (void);

void mandel_tile (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      image[i * image_dim + j] = compute_one_pixel (i, j);
}

void do_tile (int x, int y, int width, int height)
{
  int me = omp_get_thread_num ();

  ezm_start_work (recorder, me);

  mandel_tile (x, y, width, height);

  ezm_end_2D (recorder, me, x, y, width, height);
}

void mandel_compute (void)
{
  ezm_start_iteration (recorder);

#pragma omp parallel for collapse(2) schedule(runtime)
  for (int y = 0; y < image_dim; y += tile_size)
    for (int x = 0; x < image_dim; x += tile_size)
      do_tile (x, y, tile_size, tile_size);

  ezm_end_iteration (recorder);

  zoom ();
}

#define ROUND_TILE_SIZE(n) (((n) + 15UL) & ~15UL)

int main (int argc, char *argv[])
{
  if (argc > 1 && strcmp (argv[1], "-t") == 0) {
    do_trace = 1;
    argc--;
    argv++;
  }

  if (argc > 1) {
    unsigned n = atoi (argv[1]);
    if (n > 0)
      tile_size = ROUND_TILE_SIZE (n);
    printf ("Tile size: %d (was: %d)\n", tile_size, n);
  }

  ezm_init (NULL, do_trace ? EZM_NO_DISPLAY : 0);

  img2d_obj_init (&img, image_dim, image_dim);

  // Allocate main data
  size_t s = img2d_obj_size (&img);
  image    = mmap (NULL, s, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
  if (image == NULL)
    exit_with_error ("Cannot allocate main image: mmap failed");

  if (!do_trace) {
    // Create main EZV context (i.e. create OpenGL window)
    ctx[nb_ctx++] =
        ezv_ctx_create (EZV_CTX_TYPE_IMG2D, "Mandel", 0, 0, WIN_WIDTH,
                        WIN_HEIGHT, EZV_ENABLE_HUD | EZV_ENABLE_PICKING);
    // Attach img
    ezv_img2d_set_img (ctx[0], &img);
    // Set data colors
    ezv_use_data_colors_predefined (ctx[0], EZV_PALETTE_RGBA_PASSTHROUGH);
    ezv_use_cpu_colors (ctx[0]);

    // Create head-up displays
    create_huds (ctx[0]);
  }

  // Profiling
  recorder = ezm_recorder_create (nb_openmp_threads (), 0);
  if (do_trace) {
    ezm_recorder_attach_tracerec (recorder, "data/traces/mandel.evt", "Mandelbrot OpenMP");
    ezm_recorder_store_img2d_dim (recorder, image_dim, image_dim);
  } else {
    ezm_set_cpu_palette (recorder, EZV_PALETTE_RAINBOW, 0);
    ezm_helper_add_perfmeter (recorder, ctx, &nb_ctx);
    ezm_helper_add_footprint (recorder, ctx, &nb_ctx);
  }
  ezm_recorder_enable (recorder, 1);

  mandel_init ();

  int it = 0;
  while (1) {
    if (it < 50) {
      // compute one image
      mandel_compute ();
      it++;
      if (!do_trace)
        ezv_set_data_colors (ctx[0], image);
    }

    if (!do_trace) {
      // check for UI events
      process_events (it >= 50);

      // refresh display
      ezv_render (ctx, nb_ctx);
    } else if (it == 50)
      break;
  }

  printf ("Completed after %d iterations\n", it);

  ezm_recorder_destroy (recorder);

  if (image)
    munmap (image, s);

  return 0;
}

////////////////////////
static unsigned nb_openmp_threads (void)
{
  unsigned n;
#pragma omp parallel
  {
#pragma omp single
    n = omp_get_num_threads ();
  }
  return n;
}

// Mandel data
#define MAX_ITERATIONS 2048
#define ZOOM_SPEED -0.01

float mandel_leftX   = -0.2395;
float mandel_rightX  = -0.2275;
float mandel_topY    = .660;
float mandel_bottomY = .648;

float mandel_xstep;
float mandel_ystep;

static void mandel_init (void)
{
  mandel_xstep = (mandel_rightX - mandel_leftX) / image_dim;
  mandel_ystep = (mandel_topY - mandel_bottomY) / image_dim;
}

static uint32_t iteration_to_color (unsigned iter)
{
  uint8_t r = 0, g = 0, b = 0;

  if (iter < MAX_ITERATIONS) {
    if (iter < 64) {
      r = iter * 2; /* 0x0000 to 0x007E */
    } else if (iter < 128) {
      r = (((iter - 64) * 128) / 126) + 128; /* 0x0080 to 0x00C0 */
    } else if (iter < 256) {
      r = (((iter - 128) * 62) / 127) + 193; /* 0x00C1 to 0x00FF */
    } else if (iter < 512) {
      r = 255;
      g = (((iter - 256) * 62) / 255) + 1; /* 0x01FF to 0x3FFF */
    } else if (iter < 1024) {
      r = 255;
      g = (((iter - 512) * 63) / 511) + 64; /* 0x40FF to 0x7FFF */
    } else if (iter < 2048) {
      r = 255;
      g = (((iter - 1024) * 63) / 1023) + 128; /* 0x80FF to 0xBFFF */
    } else {
      r = 255;
      g = (((iter - 2048) * 63) / 2047) + 192; /* 0xC0FF to 0xFFFF */
    }
  }
  return ezv_rgb (r, g, b);
}

static void zoom (void)
{
  float xrange = (mandel_rightX - mandel_leftX);
  float yrange = (mandel_topY - mandel_bottomY);

  mandel_leftX += ZOOM_SPEED * xrange;
  mandel_rightX -= ZOOM_SPEED * xrange;
  mandel_topY -= ZOOM_SPEED * yrange;
  mandel_bottomY += ZOOM_SPEED * yrange;

  mandel_xstep = (mandel_rightX - mandel_leftX) / image_dim;
  mandel_ystep = (mandel_topY - mandel_bottomY) / image_dim;
}

static uint32_t compute_one_pixel (int i, int j)
{
  float cr = mandel_leftX + mandel_xstep * j;
  float ci = mandel_topY - mandel_ystep * i;
  float zr = 0.0, zi = 0.0;

  int iter;

  // Pour chaque pixel, on calcule les termes d'une suite, et on
  // s'arrête lorsque |Z| > 2 ou lorsqu'on atteint MAX_ITERATIONS
  for (iter = 0; iter < MAX_ITERATIONS; iter++) {
    float rc = zr * zr;
    float ic = zi * zi;

    /* Stop iterations when |Z| > 2 */
    if (rc + ic > 4.0)
      break;

    float xy = zr * zi;
    /* Z = Z^2 + C */
    zr = (cr - ic) + rc;
    zi = 2.0 * xy + ci;
  }

  return iteration_to_color (iter);
}
