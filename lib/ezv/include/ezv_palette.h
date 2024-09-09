#ifndef EZV_PALETTE_H
#define EZV_PALETTE_H

#define MAX_PALETTE 4096

typedef enum
{
  EZV_PALETTE_UNDEFINED,
  EZV_PALETTE_RGBA_PASSTHROUGH,
  EZV_PALETTE_CUSTOM,
  EZV_PALETTE_LINEAR,
  EZV_PALETTE_HEAT,
  EZV_PALETTE_3GAUSS,
  EZV_PALETTE_LIFE,
  EZV_PALETTE_BARBIE_KEN,
  EZV_PALETTE_CHRISTMAS,
  EZV_PALETTE_YELLOW,
  EZV_PALETTE_RAINBOW,
} ezv_palette_name_t;

typedef struct
{
  ezv_palette_name_t name;
  unsigned max_colors;
  float *colors; // 4 floats per color
} ezv_palette_t;

void ezv_palette_init (ezv_palette_t *palette);
void ezv_palette_delete (ezv_palette_t *palette);

void ezv_palette_set_RGBA_passthrough (ezv_palette_t *palette);
void ezv_palette_set_raw (ezv_palette_t *palette, float *data,
                             unsigned size);
void ezv_palette_set_from_RGBAi (ezv_palette_t *palette,
                                    unsigned colors[], unsigned size);
void ezv_palette_set_predefined (ezv_palette_t *palette,
                                    ezv_palette_name_t name);

int ezv_palette_is_defined (ezv_palette_t *palette);


#endif
