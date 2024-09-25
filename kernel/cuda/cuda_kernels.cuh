#ifndef EASYPAP_CUDA_KERNELS_CUH
#define EASYPAP_CUDA_KERNELS_CUH

#include <stdint.h>

#define get_i() (blockIdx.y * blockDim.y + threadIdx.y)
#define get_j() (blockIdx.x * blockDim.x + threadIdx.x)
#define get_index() ((get_i ()) * DIM + (get_j ()))

#if __BYTE_ORDER == __LITTLE_ENDIAN

// Color to red
__device__ static inline uint8_t extract_red (uint32_t c)
{
  return (uint8_t)c;
}

// Color to green
__device__ static inline uint8_t extract_green (uint32_t c)
{
  return (uint8_t)(c >> 8);
}

// Color to blue
__device__ static inline uint8_t extract_blue (uint32_t c)
{
  return (uint8_t)(c >> 16);
}

// Color to alpha
__device__ static inline uint8_t extract_alpha (uint32_t c)
{
  return (uint8_t)(c >> 24);
}

// Make a color from red, green, blue and alpha
__device__ static inline uint32_t rgba (uint8_t r, uint8_t g, uint8_t b,
                                        uint8_t a)
{
  return ((uint32_t)r) | (((uint32_t)g) << 8) | (((uint32_t)b) << 16) |
         (((uint32_t)a) << 24);
}

// Make a color from red, green and blue
__device__ static inline uint32_t rgb (uint8_t r, uint8_t g, uint8_t b)
{
  return rgba (r, g, b, 255);
}

// RGB invert mask
__device__ static inline unsigned rgb_invert_mask (void)
{
  return 0x00FFFFFF;
}

#elif __BYTE_ORDER == __BIG_ENDIAN

// Color to red
__device__ static inline uint8_t extract_red (uint32_t c)
{
  return (uint8_t)(c >> 24);
}

// Color to green
__device__ static inline uint8_t extract_green (uint32_t c)
{
  return (uint8_t)(c >> 16);
}

// Color to blue
__device__ static inline uint8_t extract_blue (uint32_t c)
{
  return (uint8_t)(c >> 8);
}

// Color to alpha
__device__ static inline uint8_t extract_alpha (uint32_t c)
{
  return (uint8_t)c;
}

// Make a color from red, green, blue and alpha
__device__ static inline uint32_t rgba (uint8_t r, uint8_t g, uint8_t b,
                                        uint8_t a)
{
  return (((uint32_t)r) << 24) | (((uint32_t)g) << 16) | (((uint32_t)b) << 8) |
         ((uint32_t)a);
}

// Make a color from red, green and blue
__device__ static inline uint32_t rgb (uint8_t r, uint8_t g, uint8_t b)
{
  return rgba (r, g, b, 255);
}

// RGB invert mask
__device__ static inline unsigned rgb_invert_mask (void)
{
  return 0xFFFFFF00;
}

#else
#error Failed to determine endianness
#endif

#endif // EASYPAP_CUDA_KERNELS_CUH
