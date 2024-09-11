#ifndef COMMON_IS_DEF
#define COMMON_IS_DEF
//
// !!! DO NOT MODIFY THIS FILE !!!
//
// Utility functions for OpenCL
//

#ifndef DIM
#define DIM 1
#define TILE_W 1
#define TILE_H 1
#define PARAM 1
#endif

#ifdef IS_LITTLE_ENDIAN

static inline unsigned rgb_mask (void)
{
  return 0x00FFFFFF;
}

// Color to component

// Color to red
static inline uchar c2r (unsigned c)
{
  return (uchar)c;
}

// Color to green
static inline uchar c2g (unsigned c)
{
  return (uchar)(c >> 8);
}

// Color to blue
static inline uchar c2b (unsigned c)
{
  return (uchar)(c >> 16);
}

// Color to alpha
static inline uchar c2a (unsigned c)
{
  return (uchar)(c >> 24);
}

// Component to color

// Red to color
static inline unsigned r2c (uchar r)
{
  return (unsigned)r;
}

// Green to color
static inline unsigned g2c (uchar g)
{
  return ((unsigned)g) << 8;
}

// Blue to color
static inline unsigned b2c (uchar b)
{
  return ((unsigned)b) << 16;
}

// Alpha to color
static inline unsigned a2c (uchar a)
{
  return ((unsigned)a) << 24;
}

// color to vector
static uchar4 color_to_char4 (unsigned c)
{
  return (*((uchar4 *) &c));
}

// vector to color
static unsigned char4_to_color (uchar4 v)
{
  return *((unsigned *) &v);
}

#else // IS_BIG_ENDIAN

static inline unsigned rgb_mask (void)
{
  return 0xFFFFFF00;
}

// Color to component

// Color to red
static inline uchar c2r (unsigned c)
{
  return (uchar)(c >> 24);
}

// Color to green
static inline uchar c2g (unsigned c)
{
  return (uchar)(c >> 16);
}

// Color to blue
static inline uchar c2b (unsigned c)
{
  return (uchar)(c >> 8);
}

// Color to alpha
static inline uchar c2a (unsigned c)
{
  return (uchar)c;
}

// Component to color

// Red to color
static inline unsigned r2c (uchar r)
{
  return ((unsigned)r) << 24;
}

// Green to color
static inline unsigned g2c (uchar g)
{
  return ((unsigned)g) << 16;
}

// Blue to color
static inline unsigned b2c (uchar b)
{
  return ((unsigned)b) << 8;
}

// Alpha to color
static inline unsigned a2c (uchar a)
{
  return (unsigned)a;
}

// color to vector
static uchar4 color_to_char4 (unsigned c)
{
  return (*((uchar4 *) &c)).s3210;
}

// vector to color
static unsigned char4_to_color (uchar4 v)
{
  uchar4 v2 = v.s3210;
  return *((unsigned *) &v2);
}

#endif


// Build color from red, green, blue and alpha (RGBA) components
static inline unsigned rgba (uchar r, uchar g, uchar b, uchar a)
{
  return r2c (r) | g2c (g) | b2c (b) | a2c (a);
}

// Build color from red, green and blue (RGB) components
static inline unsigned rgb (uchar r, uchar g, uchar b)
{
  return rgba (r, g, b, 255);
}

static int4 color_to_int4 (unsigned c)
{
  return convert_int4 (color_to_char4 (c));
}

static unsigned int4_to_color (int4 v)
{
  return char4_to_color (convert_uchar4 (v));
}

static float4 color_to_float4 (unsigned c)
{
  return convert_float4 (color_to_char4 (c)) / 255.0f;
}

static unsigned float4_to_color (float4 v)
{
  return char4_to_color (convert_uchar4 (v * 255.0f));
}

#ifdef GL_BUFFER_SHARING

// This is a generic version of a kernel updating the OpenGL texture buffer.
// It should work with most of existing kernels.
// Can be refined as update_texture_<kernel>
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);

  write_imagef (tex, (int2)(x, y), color_to_float4 (cur [y * DIM + x]));
}

#endif

__kernel void bench_kernel (void)
{
}

#define extract_red(c) c2r (c)
#define extract_green(c) c2g (c)
#define extract_blue(c) c2b (c)
#define extract_alpha(c) c2a (c)

#endif