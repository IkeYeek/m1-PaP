#include <ezv_rgba.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "ezv_palette.h"

static void check_redef (ezv_palette_t *palette)
{
  if (ezv_palette_is_defined (palette))
    exit_with_error ("Attempt to redefine existing palette");
}

static void alloc_palette (ezv_palette_t *palette, unsigned size)
{
  if (size > MAX_PALETTE)
    exit_with_error ("Palette cannot exceed %d colors", MAX_PALETTE);

  palette->max_colors = size;
  palette->colors     = malloc (size * sizeof (vec4));
}

void ezv_palette_init (ezv_palette_t *palette)
{
  palette->name       = EZV_PALETTE_UNDEFINED;
  palette->max_colors = 0;
  palette->colors     = NULL;
}

void ezv_palette_clean (ezv_palette_t *palette)
{
  if (palette->colors != NULL)
    free (palette->colors);

  ezv_palette_init (palette);
}

int ezv_palette_is_defined (ezv_palette_t *palette)
{
  return palette->name != EZV_PALETTE_UNDEFINED;
}

unsigned ezv_palette_size (ezv_palette_t *palette)
{
  return palette->max_colors;
}

void ezv_palette_set_RGBA_passthrough (ezv_palette_t *palette)
{
  check_redef (palette);

  palette->name       = EZV_PALETTE_RGBA_PASSTHROUGH;
  palette->max_colors = 0;
  palette->colors     = NULL;
}

void ezv_palette_set_raw (ezv_palette_t *palette, float *data, unsigned size)
{
  check_redef (palette);

  alloc_palette (palette, size);

  memcpy (palette->colors, data, size * sizeof (vec4));

  palette->name = EZV_PALETTE_CUSTOM;
}

void ezv_palette_set_from_RGBAi (ezv_palette_t *palette, uint32_t colors[],
                                 unsigned size)
{
  check_redef (palette);

  alloc_palette (palette, size);

  for (int c = 0; c < size; c++) {
    palette->colors[c][0] = (float)ezv_c2r (colors[c]) / 255.;
    palette->colors[c][1] = (float)ezv_c2g (colors[c]) / 255.;
    palette->colors[c][2] = (float)ezv_c2b (colors[c]) / 255.;
    palette->colors[c][3] = (float)ezv_c2a (colors[c]) / 255.;
  }

  palette->name = EZV_PALETTE_CUSTOM;
}

void ezv_palette_set_predefined (ezv_palette_t *palette,
                                 ezv_palette_name_t name)
{
  check_redef (palette);

  switch (name) {
  case EZV_PALETTE_LINEAR: {
    vec4 colors[]  = {{0.0f, 0.0f, 1.0f, 1.0f},  // blue
                      {1.0f, 0.0f, 0.0f, 1.0f}}; // red
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_HEAT: {
    vec4 colors[]  = {{0.0f, 0.0f, 1.0f, 1.0f},  // blue
                      {0.0f, 1.0f, 1.0f, 1.0f},  // cyan
                      {0.0f, 1.0f, 0.0f, 1.0f},  // green
                      {1.0f, 1.0f, 0.0f, 1.0f},  // yellow
                      {1.0f, 0.0f, 0.0f, 1.0f}}; // red
    const int size = 5;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_3GAUSS: {
    vec4 colors[]  = {{0.0, 0.0, 1.0, 1.0},       // blue
                      {0.1667, 1.0, 0.8333, 1.0}, //
                      {0.3333, 0.0, 0.6666, 1.0}, //
                      {0.5, 1.0, 0.5, 1.0},       //
                      {0.6666, 0.0, 0.3333, 1.0}, //
                      {0.8333, 1.0, 0.1667, 1.0}, //
                      {1.0, 0.0, 0.0, 1.0}};      // red
    const int size = 7;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_RAINBOW: {
    vec4 colors[]  = {{0.0f, 0.0f, 1.0f, 1.f},  // blue
                      {0.0f, 0.5f, 1.0f, 1.f},
                      {0.0f, 1.0f, 1.0f, 1.f},  // cyan
                      {0.0f, 1.0f, 0.5f, 1.f},
                      {0.0f, 1.0f, 0.0f, 1.f},  // green
                      {0.5f, 1.0f, 0.0f, 1.f},
                      {1.0f, 1.0f, 0.0f, 1.f},  // yellow
                      {1.0f, 0.5f, 0.0f, 1.f},
                      {1.0f, 0.0f, 0.0f, 1.f},  // red
                      {1.0f, 0.0f, 0.5f, 1.f},
                      {1.0f, 0.0f, 1.0f, 1.f},  // pink
                      {0.5f, 0.0f, 1.0f, 1.f}};
    const int size = 12;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_LIFE: {
    vec4 colors[]  = {{0.1f, 0.1f, 0.1f, 1.0f},  // dark grey
                      {1.0f, 1.0f, 0.0f, 1.0f}}; // yellow
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_BARBIE_KEN: {
    vec4 colors[]  = {{0.0f, 1.0f, 1.0f, 1.0f},    // cyan
                      {1.0f, 1.0f, 0.0f, 1.0f},    // yellow
                      {0.85f, 0.0f, 0.65f, 1.0f}}; // magenta
    const int size = 3;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_CHRISTMAS: {
    vec4 colors[]  = {{0.0f, 0.3f, 0.0f, 1.0f},  // dark green
                      {1.0f, 0.0f, 0.0f, 1.0f}}; // red
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_YELLOW: {
    vec4 colors[]  = {{1.0f, 1.0f, 0.0f, 0.0f},  // yellow 0% opacity
                      {1.0f, 1.0f, 0.0f, 1.0f}}; // yellow 100% opacity
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * sizeof (vec4));
    break;
  }
  case EZV_PALETTE_EASYPAP: {
    uint32_t colors[] = {
        ezv_rgb (0xFF, 0xFF, 0x00), // Yellow
        ezv_rgb (0xFF, 0x00, 0x00), // Red
        ezv_rgb (0x00, 0xFF, 0x00), // Green
        ezv_rgb (0xAE, 0x4A, 0xFF), // Purple
        ezv_rgb (0x00, 0xFF, 0xFF), // Cyan
        ezv_rgb (0xB0, 0xB0, 0xB0), // Grey
        ezv_rgb (0x64, 0x64, 0xFF), // Blue
        ezv_rgb (0xFF, 0xBF, 0xF7), // Pale Pink
        ezv_rgb (0xFF, 0xD5, 0x91), // Cream
        ezv_rgb (0xCF, 0xFF, 0xBF), // Pale Green
        ezv_rgb (0xF0, 0x80, 0x80), // Light Coral
        ezv_rgb (0xE0, 0x00, 0xE0)  // Magenta
    };
    const int size = 12;

    ezv_palette_set_from_RGBAi (palette, colors, size);
    break;
  }
  case EZV_PALETTE_CUSTOM: {
    exit_with_error ("Custom data palette must be configured with "
                     "'ezv_use_data_colors'");
  }
  case EZV_PALETTE_RGBA_PASSTHROUGH: {
    // Odd way of configuring RGBA, but why not
    ezv_palette_set_RGBA_passthrough (palette);
    break;
  }
  case EZV_PALETTE_UNDEFINED: {
    // Nothing to do
    break;
  }
  default:
    exit_with_error ("Unsupported palette type (%d)", name);
  }

  palette->name = name;
}

uint32_t ezv_palette_get_color_from_value (ezv_palette_t *palette, float value)
{
  const unsigned size = palette->max_colors;
  float scale         = value * (float)(size - 1);
  int ind             = scale;
  float frac;

  if (ind < size - 1)
    frac = scale - ind;
  else {
    frac = 1.0;
    ind--;
  }

  vec4 color;
  glm_vec4_mix (palette->colors[ind], palette->colors[ind + 1], frac, color);
  glm_vec4_scale (color, 255.0, color);

  return ezv_rgba (color[0], color[1], color[2], color[3]);
}

uint32_t ezv_palette_get_color_from_index (ezv_palette_t *palette,
                                           unsigned index)
{
  vec4 color;

  memcpy (color, palette->colors[index % palette->max_colors], sizeof (vec4));
  glm_vec4_scale (color, 255.0, color);

  return ezv_rgba (color[0], color[1], color[2], color[3]);
}
