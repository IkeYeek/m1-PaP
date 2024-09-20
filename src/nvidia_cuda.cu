#include <cstdio>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "cppdefs.h"
EXTERN
{
#include "constants.h"
#include "debug.h"
#include "global.h"
#include "hooks.h"
#include "img_data.h"
#include "ezp_ctx.h"
}

#define CHECK_CUDA_ERROR(ret)                                                  \
  if (ret != cudaSuccess) {                                                    \
    fprintf (stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString (ret),    \
             __FILE__, __LINE__);                                              \
    exit (EXIT_FAILURE);                                                       \
  }

GLuint texid;
cudaGraphicsResource_t texResource;
cudaSurfaceObject_t surfaceObject;
uint32_t *gpu_image, *gpu_alt_image;

dim3 grid, block;

int nb_gpu;
int nb_threads_per_block;

static void cuda_show_devices (void)
{

  cudaError_t ret;
  int nb_devices;
  ret = cudaGetDeviceCount (&nb_devices);
  CHECK_CUDA_ERROR (ret);

  printf ("Number of devices: %d\n", nb_devices);

  for (int i = 0; i < nb_devices; i++) {
    cudaDeviceProp props;
    ret = cudaGetDeviceProperties (&props, i);
    CHECK_CUDA_ERROR (ret);

    printf ("Device %d: [%s]\n", i, props.name);
    printf ("\tGlobal memory: %.2f GB\n",
            (double)props.totalGlobalMem / 1024 / 1024 / 1024);
    printf ("\tConstant memory: %lu KB\n", props.totalConstMem / 1024);
    printf ("\tShared memory per block: %lu KB\n",
            props.sharedMemPerBlock / 1024);
    printf ("\tRegisters per block: %d\n", props.regsPerBlock);
    printf ("\tThreads per block: %d\n", props.maxThreadsPerBlock);
    printf ("\tNumber of SM: %d\n", props.multiProcessorCount);
    printf ("\tFrequency: %d MHz\n", props.clockRate / 1000);
  }
}

EXTERN unsigned easypap_number_of_gpus_cuda (void)
{
  return (gpu_used ? 1 : 0);
}

EXTERN void cuda_send_data (void)
{
  cudaError_t ret = cudaMemcpy (gpu_image, image, DIM * DIM * sizeof (uint32_t),
                                cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR (ret);
}

EXTERN void cuda_retrieve_data (void)
{
  cudaError_t ret = cudaMemcpy (image, gpu_image, DIM * DIM * sizeof (uint32_t),
                                cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR (ret);
}

EXTERN void cuda_init (int show_config, int silent)
{

  if (show_config) {
    cuda_show_devices ();
    exit (0);
  }

  cudaError_t ret;

  // get number of devices
  ret = cudaGetDeviceCount (&nb_gpu);
  CHECK_CUDA_ERROR (ret);

  int id_gpu = 0;
  char *str  = getenv ("DEVICE");
  if (str != NULL)
    id_gpu = atoi (str);

  if (id_gpu >= nb_gpu || id_gpu < 0) {
    fprintf (stderr, "Error: GPU %d does not exist\n", id_gpu);
    exit (EXIT_FAILURE);
  }

  ret = cudaSetDevice (id_gpu);
  CHECK_CUDA_ERROR (ret);

  cudaDeviceProp props;
  ret = cudaGetDeviceProperties (&props, id_gpu);
  CHECK_CUDA_ERROR (ret);
  nb_threads_per_block = props.maxThreadsPerBlock;

  if (TILE_W * TILE_H > nb_threads_per_block) {
    fprintf (stderr, "Error: TILE_SIZE > %d\n", nb_threads_per_block);
    exit (EXIT_FAILURE);
  }

  printf ("Using GPU %d: [%s]\n", id_gpu, props.name);
}

EXTERN unsigned cuda_compute (unsigned nb_iter)
{

  cudaError_t ret;

  // call kernel
  for (int i = 0; i < nb_iter; i++) {

    the_cuda_kernel<<<grid, block>>> (gpu_image, gpu_alt_image, DIM);
    ret = cudaDeviceSynchronize ();
    CHECK_CUDA_ERROR (ret);
    if (the_cuda_kernel_finish != NULL) {
      the_cuda_kernel_finish<<<1, 1>>> (DIM);
      ret = cudaDeviceSynchronize ();
      CHECK_CUDA_ERROR (ret);
    }

    // swap images
    uint32_t *tmp = gpu_image;
    gpu_image     = gpu_alt_image;
    gpu_alt_image = tmp;
  }

  return 0;
}

EXTERN void cuda_establish_bindings (void)
{
  the_compute = (int_func_t) bind_it (kernel_name, (char *)"cuda_invoke", variant_name, 0);
  if (the_compute == NULL) {
    the_compute = cuda_compute;
    PRINT_DEBUG ('c', "Using the generic CUDA kernel launcher\n");
  }
  if (variant_name == NULL || strcmp(variant_name, "") == 0)
    the_cuda_kernel = (cuda_kernel_func_t) bind_it (kernel_name, (char *)"cuda",
                                                    (char *)"/", 1); // "/" for not find "kernel_name_cuda_"
  else
    the_cuda_kernel = (cuda_kernel_func_t) bind_it (kernel_name, (char *)"cuda",
                                                    variant_name, 2);
  the_cuda_kernel_finish = (cuda_kernel_finish_func_t) bind_it
    (kernel_name, (char *)"cuda_finish", variant_name, 0);
}

static void cuda_map_textures (unsigned tex)
{
  texid = tex;

  cudaError_t ret;

  // register texture with CUDA
  ret = cudaGraphicsGLRegisterImage (&texResource, texid, GL_TEXTURE_2D,
                                     cudaGraphicsRegisterFlagsSurfaceLoadStore);
  CHECK_CUDA_ERROR (ret);

  ret = cudaGraphicsMapResources (1, &texResource);
  CHECK_CUDA_ERROR (ret);

  // Récupération d'un cudaArray représentant la texture
  cudaArray_t array;
  ret = cudaGraphicsSubResourceGetMappedArray (&array, texResource, 0, 0);
  CHECK_CUDA_ERROR (ret);

  // Création d'un objet de surface à partir du cudaArray
  cudaResourceDesc resDesc;
  memset (&resDesc, 0, sizeof (resDesc));
  resDesc.resType         = cudaResourceTypeArray;
  resDesc.res.array.array = array;

  ret = cudaCreateSurfaceObject (&surfaceObject, &resDesc);
  CHECK_CUDA_ERROR (ret);

  // ret = cudaGraphicsMapResources(1, &texResource);
  // CHECK_CUDA_ERROR(ret);

  // note : we should call somewhere
  // cudaGraphicsUnmapResources(1, &texResource)
}

EXTERN void cuda_alloc_buffers (void)
{
  cudaError_t ret;
  ret = cudaMalloc ((void **)&gpu_image, DIM * DIM * sizeof (uint32_t));
  CHECK_CUDA_ERROR (ret);
  ret = cudaMalloc ((void **)&gpu_alt_image, DIM * DIM * sizeof (uint32_t));
  CHECK_CUDA_ERROR (ret);

  if (do_display && easypap_gl_buffer_sharing) {
    int gl_buffer_ids[1];
    ezv_switch_to_context (ctx[0]);
    ezv_get_shareable_buffer_ids (ctx[0], gl_buffer_ids);

    cuda_map_textures (gl_buffer_ids[0]);
  }
}

EXTERN void cuda_build_program (int list_variants)
{
  if (list_variants) {
    printf ("currently no variants for cuda\n");
    exit (0);
  }

  grid.x = DIM / TILE_W;
  grid.y = DIM / TILE_H;
  grid.z = 1;

  block.x = TILE_W;
  block.y = TILE_H;
  block.z = 1;

  printf ("Tile size: %dx%d\n", TILE_W, TILE_H);
}

__global__ void cuda_update_texture (cudaSurfaceObject_t target,
                                     unsigned *image, unsigned dim)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dim && y < dim) {

    unsigned pixel = image[y * dim + x];
    // pixel to uchar4

    // uchar4 data = make_uchar4(pixel >> 24, (pixel >> 16) & 255, (pixel >> 8)
    // & 255, pixel & 255);

    uchar4 data = make_uchar4 (pixel & 255, (pixel >> 8) & 255,
                               (pixel >> 16) & 255, pixel >> 24);

    surf2Dwrite (data, target, x * sizeof (uchar4), y);
  }
}

EXTERN void cuda_update_texture (void)
{

  cudaError_t ret;

  // cudaGraphicsMapResources(1, &texResource);

  cuda_update_texture<<<grid, block>>> (surfaceObject, gpu_image, DIM);
  ret = cudaDeviceSynchronize ();
  CHECK_CUDA_ERROR (ret);

  // cudaGraphicsUnmapResources(1, &texResource);
}
