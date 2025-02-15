#ifndef IMG_DATA_IS_DEF
#define IMG_DATA_IS_DEF

#include "global.h"
#include "cppdefs.h"
#include "ezv.h"

#include <stdint.h>

// Images are DIM * DIM arrays of pixels
// Tiles have a size of CPU_TILE_H * CPU_TILE_W
// An image contains CPU_NBTILES_Y * CPU_NBTILES_X

extern unsigned DIM;

extern unsigned TILE_W;
extern unsigned TILE_H;
extern unsigned NB_TILES_X;
extern unsigned NB_TILES_Y;

extern unsigned GPU_SIZE_X;
extern unsigned GPU_SIZE_Y;

extern uint32_t *RESTRICT image, *RESTRICT alt_image;

static inline uint32_t *img_cell (uint32_t *RESTRICT i, int l, int c)
{
  return i + l * DIM + c;
}

#define cur_img(y, x) (*img_cell (image, (y), (x)))
#define next_img(y, x) (*img_cell (alt_image, (y), (x)))

static inline void swap_images (void)
{
  uint32_t *tmp = image;

  image     = alt_image;
  alt_image = tmp;
}

void img_data_init (void);
void img_data_alloc (void);
void img_data_free (void);
void img_data_imgload (void);
void img_data_replicate (void);

extern img2d_obj_t easypap_img_desc;

void img_data_set_default_palette_if_none_defined (void);
void img_data_init_huds (int show);
void img_data_refresh (unsigned iter);
void img_data_dump_to_file (char *filename);
void img_data_save_thumbnail (unsigned iteration);
void img_data_do_pick (void);

// Useful color functions

unsigned heat_to_rgb (float v); // 0.0 = cold, 1.0 = hot
unsigned heat_to_3gauss_rgb (double v); // 0.0 = cold, 1.0 = hot

#endif
