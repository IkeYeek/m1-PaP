
#include <stdlib.h>

#include "trace_colors.h"
#include "ezv_palette.h"
#include "ezv_rgba.h"

unsigned TRACE_MAX_COLORS = 0;

static uint32_t *cpu_colors = NULL;

uint32_t trace_cpu_color (int pu)
{
  return cpu_colors[pu];
}

void trace_colors_init (unsigned npus)
{
  ezv_palette_t palette;
  int cyclic = 1;

  cpu_colors = malloc ((npus + 1) * sizeof (uint32_t));
  TRACE_MAX_COLORS = npus;

  ezv_palette_init (&palette);
  ezv_palette_set_predefined (&palette, EZV_PALETTE_EASYPAP);

  for (int c = 0; c < npus; c++)
    if (cyclic)
      cpu_colors[c] = ezv_palette_get_color_from_index (&palette, c);
    else
      cpu_colors[c] = ezv_palette_get_color_from_value (&palette, (float)c / (float)(npus - 1));

  cpu_colors[TRACE_MAX_COLORS] = ezv_rgb (255, 255, 255); // White
}
