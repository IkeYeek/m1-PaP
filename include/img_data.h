#ifndef IMG_DATA_IS_DEF
#define IMG_DATA_IS_DEF

#include "global.h"
#include "cppdefs.h"

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

void img_data_alloc (void);
void img_data_free (void);
void img_data_replicate (void);

// Useful color functions

static inline int extract_red (uint32_t c)
{
  return c >> 24;
}

static inline int extract_green (uint32_t c)
{
  return (c >> 16) & 255;
}

static inline int extract_blue (uint32_t c)
{
  return (c >> 8) & 255;
}

static inline int extract_alpha (uint32_t c)
{
  return c & 255;
}

static inline uint32_t rgba (int r, int g, int b, int a)
{
  return (r << 24) | (g << 16) | (b << 8) | a;
}

unsigned hsv_to_rgb (float h, float s, float v);
unsigned heat_to_rgb (float h); // 0.0 = cold, 1.0 = hot
unsigned heat_to_3gauss_rgb (double v); // 0.0 = cold, 1.0 = hot

#endif
