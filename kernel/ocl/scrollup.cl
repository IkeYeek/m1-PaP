#include "kernel/ocl/common.cl"


__kernel void scrollup_ocl (__global unsigned *in, __global unsigned *out)
{
  unsigned y = get_global_id (1);
  unsigned x = get_global_id (0);
  unsigned ysource = (y == get_global_size (1) - 1 ? 0 : y + 1);
  unsigned couleur;

  couleur = in [ysource * DIM + x];

  out [y * DIM + x] = couleur;
}

__kernel void scrollup_ocl_ouf (__global unsigned *ina, __global unsigned *inb, __global unsigned *out, __global unsigned *mask, unsigned framecolor)
{
  unsigned y = get_global_id (1);
  unsigned x = get_global_id (0);
  unsigned ysource = (y == get_global_size (1) - 1 ? 0 : y + 1);
  unsigned pixel_color;
  unsigned m = mask [y * DIM + x];
  float4 color_a, color_b;
  float ratio = extract_alpha (m) / 255.0;

  pixel_color = inb [y * DIM + x] = ina [ysource * DIM + x];

  color_a = color_to_float4 (pixel_color);
  color_b = color_to_float4 (framecolor);

  out [y * DIM + x] = float4_to_color (color_a * ratio + color_b * (1.0f - ratio));
}
