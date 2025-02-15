#ifndef EASYPAP_CUDA_KERNELS_H
#define EASYPAP_CUDA_KERNELS_H

#include <cuda.h>
#include <stdint.h>

#include "ezp_gpu_event.h"

// Index manipulation macros
#define gpu_get_row() (blockIdx.y * blockDim.y + threadIdx.y)
#define gpu_get_col() (blockIdx.x * blockDim.x + threadIdx.x)
#define gpu_get_index() ((gpu_get_row ()) * DIM + (gpu_get_col ()))

typedef struct
{
  int device;
  uint32_t *curb, *nextb; // for 2Dimg
  float *curd, *nextd;    // for 3Dmesh
  cudaStream_t stream;
} cuda_gpu_t;

extern cuda_gpu_t cuda_gpu[];
extern unsigned cuda_nb_gpus;

#define cuda_device(g) cuda_gpu[g].device
#define cuda_stream(g) cuda_gpu[g].stream

#define cuda_cur_buffer(g) cuda_gpu[g].curb
#define cuda_next_buffer(g) cuda_gpu[g].nextb

#define cuda_cur_data(g) cuda_gpu[g].curd
#define cuda_next_data(g) cuda_gpu[g].nextd

extern unsigned cuda_peer_access_enabled (int device0, int device1);
extern void cuda_configure_peer_access (int device0, int device1);

#define check(err, format, ...)                                                \
  do {                                                                         \
    if (err != cudaSuccess)                                                    \
      exit_with_error (format " [CUDA error: %s]", ##__VA_ARGS__,              \
                       cudaGetErrorString (err));                              \
  } while (0)

#if __BYTE_ORDER == __LITTLE_ENDIAN

__device__ static inline unsigned rgb_mask (void)
{
  return 0x00FFFFFF;
}

// Color to component

// Color to red
__device__ static inline uint8_t c2r (unsigned c)
{
  return (uint8_t)c;
}

// Color to green
__device__ static inline uint8_t c2g (unsigned c)
{
  return (uint8_t)(c >> 8);
}

// Color to blue
__device__ static inline uint8_t c2b (unsigned c)
{
  return (uint8_t)(c >> 16);
}

// Color to alpha
__device__ static inline uint8_t c2a (unsigned c)
{
  return (uint8_t)(c >> 24);
}

// Component to color

// Red to color
__device__ static inline unsigned r2c (uint8_t r)
{
  return (unsigned)r;
}

// Green to color
__device__ static inline unsigned g2c (uint8_t g)
{
  return ((unsigned)g) << 8;
}

// Blue to color
__device__ static inline unsigned b2c (uint8_t b)
{
  return ((unsigned)b) << 16;
}

// Alpha to color
__device__ static inline unsigned a2c (uint8_t a)
{
  return ((unsigned)a) << 24;
}

// color to vector
__device__ static uchar4 color_to_char4 (unsigned c)
{
  return (*((uchar4 *) &c));
}

// vector to color
__device__ static unsigned char4_to_color (uchar4 v)
{
  return *((unsigned *) &v);
}

#else // IS_BIG_ENDIAN

__device__ static inline unsigned rgb_mask (void)
{
  return 0xFFFFFF00;
}

// Color to component

// Color to red
__device__ static inline uint8_t c2r (unsigned c)
{
  return (uint8_t)(c >> 24);
}

// Color to green
__device__ static inline uint8_t c2g (unsigned c)
{
  return (uint8_t)(c >> 16);
}

// Color to blue
__device__ static inline uint8_t c2b (unsigned c)
{
  return (uint8_t)(c >> 8);
}

// Color to alpha
__device__ static inline uint8_t c2a (unsigned c)
{
  return (uint8_t)c;
}

// Component to color

// Red to color
__device__ static inline unsigned r2c (uint8_t r)
{
  return ((unsigned)r) << 24;
}

// Green to color
__device__ static inline unsigned g2c (uint8_t g)
{
  return ((unsigned)g) << 16;
}

// Blue to color
__device__ static inline unsigned b2c (uint8_t b)
{
  return ((unsigned)b) << 8;
}

// Alpha to color
__device__ static inline unsigned a2c (uint8_t a)
{
  return (unsigned)a;
}

// color to vector
__device__ static uchar4 color_to_char4 (unsigned c)
{
  return (*((uchar4 *) &c)).s3210;
}

// vector to color
__device__ static unsigned char4_to_color (uchar4 v)
{
  uchar4 v2 = v.s3210;
  return *((unsigned *) &v2);
}

#endif


// Build color from red, green, blue and alpha (RGBA) components
__device__ static inline unsigned rgba (uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
  return r2c (r) | g2c (g) | b2c (b) | a2c (a);
}

// Build color from red, green and blue (RGB) components
__device__ static inline unsigned rgb (uint8_t r, uint8_t g, uint8_t b)
{
  return rgba (r, g, b, 255);
}

__device__ static float4 color_to_float4 (unsigned c)
{
  return make_float4 (c2r (c) / 255.f, c2g (c) / 255.0f,
                      c2b (c) / 255.0f, c2a (c) / 255.0f);
}

__device__ static unsigned float4_to_color (float4 v)
{
  return rgba (v.x * 255.0f, v.y * 255.0f, v.z * 255.0f, v.w * 255.0f);
}

__device__ static int4 color_to_int4 (unsigned c)
{
  return make_int4 (c2r (c), c2g (c), c2b (c), c2a (c));
}

__device__ static unsigned int4_to_color (int4 v)
{
  return rgba (v.x, v.y, v.z, v.w);
}


#endif // EASYPAP_CUDA_KERNELS_H
