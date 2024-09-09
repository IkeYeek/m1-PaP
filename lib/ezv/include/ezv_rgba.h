#ifndef EZV_RGBA_H
#define EZV_RGBA_H

#include <stdint.h>

#if __BYTE_ORDER == __LITTLE_ENDIAN

static inline uint32_t ezv_rgb_mask (void)
{
  return 0x00FFFFFF;
}

static inline uint32_t ezv_red_mask (void)
{
  return 0x000000FF;
}

static inline uint32_t ezv_green_mask (void)
{
  return 0x0000FF00;
}

static inline uint32_t ezv_blue_mask (void)
{
  return 0x00FF0000;
}

static inline uint32_t ezv_alpha_mask (void)
{
  return 0xFF000000;
}

// Color to component

// Color to red
static inline uint8_t ezv_c2r (uint32_t c)
{
  return (uint8_t)c;
}

// Color to green
static inline uint8_t ezv_c2g (uint32_t c)
{
  return (uint8_t)(c >> 8);
}

// Color to blue
static inline uint8_t ezv_c2b (uint32_t c)
{
  return (uint8_t)(c >> 16);
}

// Color to alpha
static inline uint8_t ezv_c2a (uint32_t c)
{
  return (uint8_t)(c >> 24);
}

// Component to color

// Red to color
static inline uint32_t ezv_r2c (uint8_t r)
{
  return (uint32_t)r;
}

// Green to color
static inline uint32_t ezv_g2c (uint8_t g)
{
  return ((uint32_t)g) << 8;
}

// Blue to color
static inline uint32_t ezv_b2c (uint8_t b)
{
  return ((uint32_t)b) << 16;
}

// Alpha to color
static inline uint32_t ezv_a2c (uint8_t a)
{
  return ((uint32_t)a) << 24;
}

#elif __BYTE_ORDER == __BIG_ENDIAN

static inline uint32_t ezv_rgb_mask (void)
{
  return 0xFFFFFF00;
}

static inline uint32_t ezv_red_mask (void)
{
  return 0xFF000000;
}

static inline uint32_t ezv_green_mask (void)
{
  return 0x00FF0000;
}

static inline uint32_t ezv_blue_mask (void)
{
  return 0x0000FF00;
}

static inline uint32_t ezv_alpha_mask (void)
{
  return 0x000000FF;
}

// Color to component

static inline uint8_t ezv_c2r (uint32_t c)
{
  return (uint8_t)(c >> 24);
}

static inline uint8_t ezv_c2g (uint32_t c)
{
  return (uint8_t)(c >> 16);
}

static inline uint8_t ezv_c2b (uint32_t c)
{
  return (uint8_t)(c >> 8);
}

static inline uint8_t ezv_c2a (uint32_t c)
{
  return (uint8_t)c;
}

// Component to color

static inline uint32_t ezv_r2c (uint8_t r)
{
  return ((uint32_t)r) << 24;
}

static inline uint32_t ezv_g2c (uint8_t g)
{
  return ((uint32_t)g) << 16;
}

static inline uint32_t ezv_b2c (uint8_t b)
{
  return ((uint32_t)b) << 8;
}

static inline uint32_t ezv_a2c (uint8_t a)
{
  return (uint32_t)a;
}

#else
#error Failed to determine endianness
#endif

// Build color from red, green, blue and alpha (RGBA) components
static inline uint32_t ezv_rgba (uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
  return ezv_r2c (r) | ezv_g2c (g) | ezv_b2c (b) | ezv_a2c (a);
}

// Build color from red, green and blue (RGB) components
static inline uint32_t ezv_rgb (uint8_t r, uint8_t g, uint8_t b)
{
  return ezv_rgba (r, g, b, 255);
}

#endif
