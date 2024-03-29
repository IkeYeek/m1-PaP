#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "mesh3d_palette.h"

#define COLORS_RGBA_PASSTHROUGH ((float *)1)

static void check_redef (mesh3d_palette_t *palette)
{
  if (mesh3d_palette_is_defined (palette))
    exit_with_error ("Attempt to redefine existing palette");
}

static void alloc_palette (mesh3d_palette_t *palette, unsigned size)
{
  if (size > MAX_PALETTE)
    exit_with_error ("Palette cannot exceed %d colors", MAX_PALETTE);

  palette->max_colors = size;
  palette->colors     = malloc (size * 4 * sizeof (float));
}

void mesh3d_palette_init (mesh3d_palette_t *palette)
{
  palette->name = MESH3D_PALETTE_CUSTOM;
  palette->max_colors = 0;
  palette->colors     = NULL;
}

int mesh3d_palette_is_defined (mesh3d_palette_t *palette)
{
  return palette->colors != NULL;
}

void mesh3d_palette_delete (mesh3d_palette_t *palette)
{
  if (palette->colors != NULL && palette->colors != COLORS_RGBA_PASSTHROUGH)
    free (palette->colors);

  mesh3d_palette_init (palette);
}

void mesh3d_palette_set_RGBA_passthrough (mesh3d_palette_t *palette)
{
  check_redef (palette);

  palette->max_colors = 0;
  palette->colors     = COLORS_RGBA_PASSTHROUGH;
}

void mesh3d_palette_set_raw (mesh3d_palette_t *palette, float *data,
                             unsigned size)
{
  check_redef (palette);

  alloc_palette (palette, size);

  memcpy (palette->colors, data, size * 4 * sizeof (float));
}

void mesh3d_palette_set_from_RGBAi (mesh3d_palette_t *palette,
                                    unsigned colors[], unsigned size)
{
  int index = 0;

  check_redef (palette);

  alloc_palette (palette, size);

  for (int c = 0; c < size; c++) {
    unsigned color           = colors[c];
    palette->colors[index++] = (color >> 24) / 255.;
    palette->colors[index++] = ((color >> 16) & 255) / 255.;
    palette->colors[index++] = ((color >> 8) & 255) / 255.;
    palette->colors[index++] = (color & 255) / 255.;
  }
}

void mesh3d_palette_set_predefined (mesh3d_palette_t *palette,
                                    mesh3d_palette_name_t name)
{
  check_redef (palette);

  switch (name) {
  case MESH3D_PALETTE_LINEAR: {
    float colors[] = {0.0f, 0.0f, 1.0f, 1.0f,  // blue
                      1.0f, 0.0f, 0.0f, 1.0f}; // red
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_HEAT: {
    float colors[] = {0.0f, 0.0f, 1.0f, 1.0f,  // blue
                      0.0f, 1.0f, 1.0f, 1.0f,  // cyan
                      0.0f, 1.0f, 0.0f, 1.0f,  // green
                      1.0f, 1.0f, 0.0f, 1.0f,  // yellow
                      1.0f, 0.0f, 0.0f, 1.0f}; // red
    const int size = 5;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_3GAUSS: {
    float colors[] = {0.0,    0.0, 1.0,    1.0,  // blue
                      0.1667, 1.0, 0.8333, 1.0,  //
                      0.3333, 0.0, 0.6666, 1.0,  //
                      0.5,    1.0, 0.5,    1.0,  //
                      0.6666, 0.0, 0.3333, 1.0,  //
                      0.8333, 1.0, 0.1667, 1.0,  //
                      1.0,    0.0, 0.0,    1.0}; // red
    const int size = 7;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_RAINBOW: {
  float colors[] = {0.0f, 0.0f, 1.0f, 1.f,  // blue
                    0.0f, 1.0f, 1.0f, 1.f,  // cyan
                    0.0f, 1.0f, 0.0f, 1.f,  // green
                    1.0f, 1.0f, 0.0f, 1.f,  // yellow
                    1.0f, 0.0f, 0.0f, 1.f,  // red
                    0.5f, 0.0f, 1.0f, 1.f,  // purple
                    1.0f, 0.0f, 1.0f, 1.f}; // pink
    const int size = 7;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_LIFE: {
    float colors[] = {0.1f, 0.1f, 0.1f, 1.0f,  // dark grey
                      1.0f, 1.0f, 0.0f, 1.0f}; // yellow
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_BARBIE_KEN: {
    float colors[] = {0.0f,  1.0f, 1.0f,  1.0f,  // cyan
                      1.0f,  1.0f, 0.0f,  1.0f,  // yellow
                      0.85f, 0.0f, 0.65f, 1.0f}; // magenta
    const int size = 3;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_CHRISTMAS: {
    float colors[] = {0.0f, 0.3f, 0.0f, 1.0f,  // dark green
                      1.0f, 0.0f, 0.0f, 1.0f}; // red
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_YELLOW: {
    float colors[] = {1.0f, 1.0f, 0.0f, 0.0f,  // yellow 0% opacity
                      1.0f, 1.0f, 0.0f, 1.0f}; // yellow 100% opacity
    const int size = 2;

    alloc_palette (palette, size);
    memcpy (palette->colors, colors, size * 4 * sizeof (float));
    break;
  }
  case MESH3D_PALETTE_CUSTOM:
    exit_with_error ("Custom data palette must be configured with "
                     "'mesh3d_configure_data_colors'");
  default:
    exit_with_error ("Unsupported palette type (%d)", name);
  }
  
  palette->name = name;
}