
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "error.h"
#include "ezv.h"

// settings
const unsigned int SCR_WIDTH  = 1024;
const unsigned int SCR_HEIGHT = 768;

#define MAX_CTX 2

static char *filename = NULL;

static unsigned *image = NULL;

static img2d_obj_t img;
static ezv_ctx_t ctx[MAX_CTX] = {NULL, NULL};
static unsigned nb_ctx        = 1;
static int pos_hud            = -1;
static int val_hud            = -1;

static void do_pick (void)
{
  int x, y;

  ezv_perform_2D_picking (ctx, nb_ctx, &x, &y);

  ezv_hud_set (ctx[0], pos_hud, "xy: (%d, %d)", x, y);

  if (x != -1 && y != -1) {
    uint32_t v = image[y * img.width + x];
    ezv_hud_set (ctx[0], val_hud, "RGBA: %02X %02X %02X %02X", ezv_c2r (v),
                 ezv_c2g (v), ezv_c2b (v), ezv_c2a (v));
  } else
    ezv_hud_set (ctx[0], val_hud, NULL);
}

static inline int get_event (SDL_Event *event, int blocking)
{
  return blocking ? SDL_WaitEvent (event) : SDL_PollEvent (event);
}

static unsigned skipped_events = 0;

static int clever_get_event (SDL_Event *event, int blocking)
{
  int r;
  static int prefetched = 0;
  static SDL_Event pr_event; // prefetched event

  if (prefetched) {
    *event     = pr_event;
    prefetched = 0;
    return 1;
  }

  r = get_event (event, blocking);

  if (r != 1)
    return r;

  // check if successive, similar events can be dropped
  if (event->type == SDL_MOUSEMOTION) {

    do {
      int ret_code = get_event (&pr_event, 0);
      if (ret_code == 1) {
        if (pr_event.type == SDL_MOUSEMOTION) {
          *event     = pr_event;
          prefetched = 0;
          skipped_events++;
        } else {
          prefetched = 1;
        }
      } else
        return 1;
    } while (prefetched == 0);
  }

  return 1;
}

static void process_events (void)
{
  SDL_Event event;

  int r = clever_get_event (&event, 1);

  if (r > 0) {
    int pick;
    ezv_process_event (ctx, nb_ctx, &event, NULL, &pick);
    if (pick)
      do_pick ();
  }
}

int main (int argc, char *argv[])
{
  ezv_init (NULL);

  if (argc > 1)
    filename = argv[1];
  else
    filename = "../../images/shibuya.png";

  img2d_obj_init_from_file (&img, filename);
  
  size_t s = img2d_obj_size (&img);
  image    = mmap (NULL, s, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
  if (image == NULL)
    exit_with_error ("Cannot allocate main image: mmap failed");

  img2d_obj_load (&img, filename, image);

  printf ("Image: width=%d, height=%d, channels=%d (%lu bytes)\n", img.width,
          img.height, img.channels, s);

  // Create SDL windows and initialize OpenGL context
  ctx[0] = ezv_ctx_create (EZV_CTX_TYPE_IMG2D, "Image", SDL_WINDOWPOS_CENTERED,
                           SDL_WINDOWPOS_UNDEFINED, SCR_WIDTH, SCR_HEIGHT,
                           EZV_ENABLE_HUD | EZV_ENABLE_PICKING);

  // Huds
  pos_hud = ezv_hud_alloc (ctx[0]);
  ezv_hud_on (ctx[0], pos_hud);
  val_hud = ezv_hud_alloc (ctx[0]);
  ezv_hud_on (ctx[0], val_hud);

  // Attach img
  ezv_img2d_set_img (ctx[0], &img);

  ezv_use_data_colors_predefined (ctx[0], EZV_PALETTE_RGBA_PASSTHROUGH);

  ezv_set_data_colors (ctx[0], image);

  // render loop
  while (1) {
    process_events ();
    ezv_render (ctx, nb_ctx);
  }

  return 0;
}
