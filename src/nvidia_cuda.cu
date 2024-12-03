#include <cstdio>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "cppdefs.h"
EXTERN
{
#include "constants.h"
#include "debug.h"
#include "error.h"
#include "ezp_ctx.h"
#include "global.h"
#include "hooks.h"
#include "img_data.h"
#include "mesh_data.h"
#include "monitoring.h"
}
#include "cuda_kernels.h"
#include "nvidia_cuda.h"

#define MESH_NEIGHBOR_ROUND 32U

typedef void (*cuda_kernel_func_t) (void);
typedef void (*cuda_kernel_2dimg_t) (unsigned *, unsigned *, unsigned);
typedef void (*cuda_kernel_3dmesh_t) (float *, float *, int *, unsigned,
                                      unsigned);

typedef void (*cuda_kernel_finish_func_t) (unsigned);
typedef void (*cuda_update_texture_func_t) (cudaSurfaceObject_t, unsigned *,
                                            unsigned);

static cuda_kernel_2dimg_t the_cuda_kernel_2dimg          = NULL;
static cuda_kernel_3dmesh_t the_cuda_kernel_3dmesh        = NULL;
static cuda_update_texture_func_t the_cuda_update_texture = NULL;

static cudaGraphicsResource_t texResource;
static cudaSurfaceObject_t surfaceObject;
static float *tex_buffer_object = NULL;

cuda_gpu_t cuda_gpu[MAX_GPU_DEVICES];
unsigned cuda_nb_gpus = 0;

static int *neighbor_soa_buffer = NULL;

static int nb_threads_per_block = 0;

unsigned GPU_SIZE_X = 0;
unsigned GPU_SIZE_Y = 0;
unsigned GPU_SIZE   = 0;
unsigned TILE       = 0;

static void cuda_show_devices (void)
{
  cudaError_t ret;
  int nb_devices;

  ret = cudaGetDeviceCount (&nb_devices);
  check (ret, "cudaGetDeviceCount");

  printf ("Number of devices: %d\n", nb_devices);

  for (int i = 0; i < nb_devices; i++) {
    cudaDeviceProp props;
    ret = cudaGetDeviceProperties (&props, i);
    check (ret, "cudaGetDeviceProperties");

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
  return (gpu_used ? cuda_nb_gpus : 0);
}

EXTERN void *cuda_alloc_host (size_t size)
{
  cudaError_t ret;
  void *ptr;

  ret = cudaMallocHost (&ptr, size);
  check (ret, "cudaMallocHost");

  return ptr;
}

EXTERN void cuda_free_host (void *ptr)
{
  cudaError_t ret;

  ret = cudaFreeHost (ptr);
  check (ret, "cudaFreeHost");
}

static void cuda_acquire (void)
{
  if (do_display && easypap_gl_buffer_sharing) {
    cudaError_t ret;

    ret = cudaGraphicsMapResources (1, &texResource, cuda_stream (0));
    check (ret, "cudaGraphicsMapResources");
  }
}

static void cuda_release (void)
{
  if (do_display && easypap_gl_buffer_sharing) {
    cudaError_t ret;

    ret = cudaGraphicsUnmapResources (1, &texResource, cuda_stream (0));
    check (ret, "cudaGraphicsUnmapResources");
  }
}

EXTERN void cuda_send_data (void)
{
  cudaError_t ret;

  if (the_send_data != NULL) {
    the_send_data ();
    PRINT_DEBUG ('i', "Init phase 7 : Initial data transferred to CUDA "
                      "device (user-defined callback)\n");
  } else if (cuda_nb_gpus == 1) {

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
      const unsigned size = DIM * DIM * sizeof (uint32_t);

      ret = cudaMemcpyAsync (cuda_cur_buffer (0), image, size,
                             cudaMemcpyHostToDevice, cuda_stream (0));
      check (ret, "cudaMemcpyAsync");
    } else {
      const unsigned size = NB_CELLS * sizeof (float);

      ret = cudaMemcpyAsync (cuda_cur_data (0), mesh_data, size,
                             cudaMemcpyHostToDevice, cuda_stream (0));
      check (ret, "cudaMemcpyAsync");
    }
  }
}

EXTERN void cuda_retrieve_data (void)
{
  cudaError_t ret;

  if (cuda_nb_gpus == 1) {

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
      const unsigned size = DIM * DIM * sizeof (uint32_t);

      ret = cudaMemcpyAsync (image, cuda_cur_buffer (0), size,
                             cudaMemcpyDeviceToHost, cuda_stream (0));
      check (ret, "cudaMemcpyAsync");
    } else {
      const unsigned size = NB_CELLS * sizeof (float);

      ret = cudaMemcpyAsync (mesh_data, cuda_cur_data (0), size,
                             cudaMemcpyDeviceToHost, cuda_stream (0));
      check (ret, "cudaMemcpyAsync");
    }

    ret = cudaStreamSynchronize (cuda_stream (0));
    check (ret, "cudaStreamSynchronize");
  }
}

static void add_gpu (int id_gpu)
{
  cudaError_t ret;

  // Important to set device *first*
  ret = cudaSetDevice (id_gpu);
  check (ret, "cudaSetDevice");

  cuda_gpu[cuda_nb_gpus].device = id_gpu;
  cudaStreamCreate (&cuda_gpu[cuda_nb_gpus].stream);

  cuda_gpu[cuda_nb_gpus].curb  = NULL;
  cuda_gpu[cuda_nb_gpus].nextb = NULL;
  cuda_gpu[cuda_nb_gpus].curd  = NULL;
  cuda_gpu[cuda_nb_gpus].nextd = NULL;

  cuda_nb_gpus++;

  cudaDeviceProp props;
  ret = cudaGetDeviceProperties (&props, id_gpu);
  check (ret, "cudaGetDeviceProperties");

  if (nb_threads_per_block == 0)
    nb_threads_per_block = props.maxThreadsPerBlock;

  PRINT_DEBUG ('c', "Using GPU %d: [%s]\n", id_gpu, props.name);
}

extern unsigned cuda_peer_access_enabled (int device0, int device1)
{
  cudaError_t ret;
  int status0 = 0, status1 = 0;

  ret = cudaDeviceCanAccessPeer (&status0, device0, device1);
  check (ret, "cudaDeviceCanAccessPeer");
  ret = cudaDeviceCanAccessPeer (&status1, device1, device0);
  check (ret, "cudaDeviceCanAccessPeer");

  return status0 & status1;
}

extern void cuda_configure_peer_access (int device0, int device1)
{
  cudaError_t ret;

  if (!cuda_peer_access_enabled (device0, device1))
    exit_with_error ("Peer access between devices %d and %d is not possible "
                     "(please call cuda_peer_access_enabled() first)",
                     device0, device1);

  cudaSetDevice (device0);
  ret = cudaDeviceEnablePeerAccess (device1, 0);
  check (ret, "cudaDeviceEnablePeerAccess");

  cudaSetDevice (device1);
  ret = cudaDeviceEnablePeerAccess (device0, 0);
  check (ret, "cudaDeviceEnablePeerAccess");
}

EXTERN void cuda_init (int show_config, int silent)
{
  cudaError_t ret;
  int total_gpus = 0;

  if (show_config) {
    cuda_show_devices ();
    exit (0);
  }

  ret = cudaGetDeviceCount (&total_gpus);
  check (ret, "cudaGetDeviceCount");

  if (use_multiple_gpu) {
    for (int id = 0; id < MIN (total_gpus, MAX_GPU_DEVICES); id++)
      add_gpu (id);
  } else { // mono-GPU mode
    int id_gpu = 0;

    char *str = getenv ("DEVICE");
    if (str != NULL)
      id_gpu = atoi (str);

    if (id_gpu >= total_gpus || id_gpu < 0)
      exit_with_error ("GPU %d does not exist\n", id_gpu);

    add_gpu (id_gpu);
  }
}

EXTERN unsigned cuda_compute_2dimg (unsigned nb_iter)
{
  cudaError_t ret;
  dim3 grid  = {GPU_SIZE_X / TILE_W, GPU_SIZE_Y / TILE_H, 1};
  dim3 block = {TILE_W, TILE_H, 1};

  ret = cudaSetDevice (cuda_device (0));
  check (ret, "cudaSetDevice");

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (int i = 0; i < nb_iter; i++) {

    the_cuda_kernel_2dimg<<<grid, block, 0, cuda_stream (0)>>> (
        cuda_cur_buffer (0), cuda_next_buffer (0), DIM);

    // swap images
    uint32_t *tmp        = cuda_cur_buffer (0);
    cuda_cur_buffer (0)  = cuda_next_buffer (0);
    cuda_next_buffer (0) = tmp;
  }

  // FIXME: should only be performed when monitoring/tracing is activated
  ret = cudaStreamSynchronize (cuda_stream (0));
  check (ret, "cudaStreamSynchronize");

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}

EXTERN unsigned cuda_compute_3dmesh (unsigned nb_iter)
{
  cudaError_t ret;
  unsigned grid  = GPU_SIZE / TILE;
  unsigned block = TILE;

  ret = cudaSetDevice (cuda_device (0));
  check (ret, "cudaSetDevice");

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (int i = 0; i < nb_iter; i++) {

    the_cuda_kernel_3dmesh<<<grid, block, 0, cuda_stream (0)>>> (
        cuda_cur_data (0), cuda_next_data (0), neighbor_soa_buffer, NB_CELLS,
        easypap_mesh_desc.max_neighbors);

    // swap data
    {
      float *tmp         = cuda_cur_data (0);
      cuda_cur_data (0)  = cuda_next_data (0);
      cuda_next_data (0) = tmp;
    }
  }

  // FIXME: should only be performed when monitoring/tracing is activated
  ret = cudaStreamSynchronize (cuda_stream (0));
  check (ret, "cudaStreamSynchronize");

  monitoring_end_tile (clock, 0, 0, NB_CELLS, 0, easypap_gpu_lane (0));

  return 0;
}

static void cuda_map_2dimg_texture (int tex)
{
  cudaError_t ret;

  if (easypap_mode != EASYPAP_MODE_2D_IMAGES)
    exit_with_error (
        "cuda_map_2dimg_texture should only be called when using 2D images");

  // register texture with CUDA
  ret = cudaGraphicsGLRegisterImage (&texResource, tex, GL_TEXTURE_2D,
                                     cudaGraphicsRegisterFlagsSurfaceLoadStore);
  check (ret, "cudaGraphicsGLRegisterImage");

  ret = cudaGraphicsMapResources (1, &texResource);
  check (ret, "cudaGraphicsMapResources");

  // Récupération d'un cudaArray représentant la texture
  cudaArray_t array;
  ret = cudaGraphicsSubResourceGetMappedArray (&array, texResource, 0, 0);
  check (ret, "cudaGraphicsSubResourceGetMappedArray");

  // Création d'un objet de surface à partir du cudaArray
  cudaResourceDesc resDesc;
  memset (&resDesc, 0, sizeof (resDesc));
  resDesc.resType         = cudaResourceTypeArray;
  resDesc.res.array.array = array;

  ret = cudaCreateSurfaceObject (&surfaceObject, &resDesc);
  check (ret, "cudaCreateSurfaceObject");

  ret = cudaGraphicsUnmapResources (1, &texResource);
  check (ret, "cudaGraphicsUnmapResources");
}

static void cuda_map_3dmesh_buffers (int tex)
{
  cudaError_t ret;
  size_t size;

  if (easypap_mode != EASYPAP_MODE_3D_MESHES)
    exit_with_error (
        "cuda_map_3dmesh_buffers should only be called when using 3D meshes");

  // register buffer with CUDA
  ret = cudaGraphicsGLRegisterBuffer (&texResource, tex,
                                      cudaGraphicsRegisterFlagsNone);
  check (ret, "cudaGraphicsGLRegisterBuffer");

  ret = cudaGraphicsMapResources (1, &texResource);
  check (ret, "cudaGraphicsMapResources");

  ret = cudaGraphicsResourceGetMappedPointer ((void **)&tex_buffer_object,
                                              &size, texResource);
  check (ret, "cudaGraphicsResourceGetMappedPointer");

  ret = cudaGraphicsUnmapResources (1, &texResource);
  check (ret, "cudaGraphicsUnmapResources");
}

EXTERN void cuda_alloc_buffers (void)
{
  cudaError_t ret;

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    const unsigned size = DIM * DIM * sizeof (uint32_t);

    for (int g = 0; g < cuda_nb_gpus; g++) {

      ret = cudaSetDevice (cuda_device (g));
      check (ret, "cudaSetDevice");

      ret = cudaMalloc ((void **)&cuda_cur_buffer (g), size);
      check (ret, "cudaMalloc");
      ret = cudaMalloc ((void **)&cuda_next_buffer (g), size);
      check (ret, "cudaMalloc");
    }

    // Optionally share texture with OpenGL
    if (do_display && easypap_gl_buffer_sharing) {
      int gl_buffer_ids[1];

      ezv_get_shareable_buffer_ids (ctx[0], gl_buffer_ids);

      cuda_map_2dimg_texture (gl_buffer_ids[0]);

      PRINT_DEBUG ('c', "OpenGL buffers shared with CUDA\n");
    }
  } else {
    const unsigned size = NB_CELLS * sizeof (float);

    // Allocate buffers inside device memory
    for (int g = 0; g < cuda_nb_gpus; g++) {

      ret = cudaSetDevice (cuda_device (g));
      check (ret, "cudaSetDevice");

      ret = cudaMalloc ((void **)&cuda_cur_data (g), size);
      check (ret, "cudaMalloc");
      ret = cudaMalloc ((void **)&cuda_next_data (g), size);
      check (ret, "cudaMalloc");
    }

    // Optionally share texture with OpenGL
    if (do_display && easypap_gl_buffer_sharing) {
      int gl_buffer_ids[1];

      ezv_get_shareable_buffer_ids (ctx[0], gl_buffer_ids);

      cuda_map_3dmesh_buffers (gl_buffer_ids[0]);

      PRINT_DEBUG ('c', "OpenGL buffers shared with CUDA\n");
    }

    if (cuda_nb_gpus == 1) {
      // Buffers hosting neighbors
      mesh_data_build_neighbors_soa (TILE); // GPU_SIZE is rounded accordingly

      const unsigned size =
          neighbor_soa_offset * easypap_mesh_desc.max_neighbors * sizeof (int);

      ret = cudaMalloc ((void **)&neighbor_soa_buffer, size);
      check (ret, "cudaMalloc");

      ret = cudaMemcpyAsync (neighbor_soa_buffer, neighbors_soa, size,
                             cudaMemcpyHostToDevice, cuda_stream (0));
      check (ret, "cudaMemcpyAsync");
    }
  }
}

EXTERN void cuda_build_program (int list_variants)
{
  char *str = NULL;

  if (list_variants) {
    printf ("Currently no variants for cuda\n");
    exit (0);
  }

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    if (!GPU_SIZE_X) {
      str = getenv ("SIZE");
      if (str != NULL)
        GPU_SIZE_X = atoi (str);
      else
        GPU_SIZE_X = DIM;

      if (GPU_SIZE_X > DIM)
        exit_with_error ("GPU_SIZE_X (%d) cannot exceed DIM (%d)", GPU_SIZE_X,
                         DIM);
    }

    if (!GPU_SIZE_Y)
      GPU_SIZE_Y = GPU_SIZE_X;

    if (GPU_SIZE_X % TILE_W)
      fprintf (stderr,
               "Warning: GPU_SIZE_X (%d) is not a multiple of TILE_W (%d)!\n",
               GPU_SIZE_X, TILE_W);

    if (GPU_SIZE_Y % TILE_H)
      fprintf (stderr,
               "Warning: GPU_SIZE_Y (%d) is not a multiple of TILE_H (%d)!\n",
               GPU_SIZE_Y, TILE_H);

    // Make sure we don't exceed the maximum group size
    if (TILE_W * TILE_H > nb_threads_per_block)
      exit_with_error ("#threads per block exceeds %d\n", nb_threads_per_block);

    printf ("Using %dx%d threads grouped in %dx%d tiles\n", GPU_SIZE_X,
            GPU_SIZE_Y, TILE_W, TILE_H);
  } else { // MESH3D
    str = getenv ("TILE");
    if (str != NULL) {
      TILE = atoi (str);
      if (TILE % 32 != 0)
        exit_with_error ("Block size (TILE) should be a multiple of 32");
    } else
      TILE = MESH_NEIGHBOR_ROUND;
    GPU_SIZE = ROUND_TO_MULTIPLE (NB_CELLS, TILE);
  }
}

__global__ void cuda_update_texture_generic (cudaSurfaceObject_t target,
                                             unsigned *image, unsigned dim)
{
  int x = gpu_get_col ();
  int y = gpu_get_row ();

  float4 data = color_to_float4 (image[y * dim + x]);

  surf2Dwrite<float4> (data, target, x * sizeof (float4), y,
                       cudaBoundaryModeClamp);
}

EXTERN void cuda_update_texture (void)
{
  cudaError_t ret;

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {

    dim3 grid  = {DIM / 32, DIM / 32, 1};
    dim3 block = {32, 32, 1};

    cuda_acquire ();

    the_cuda_update_texture<<<grid, block, 0, cuda_stream (0)>>> (
        surfaceObject, cuda_cur_buffer (0), DIM);

    ret = cudaStreamSynchronize (cuda_stream (0));
    check (ret, "cudaStreamSynchronize");

    cuda_release ();
  } else {
    const unsigned size = NB_CELLS * sizeof (float);

    cuda_acquire ();

    ret = cudaMemcpyAsync (tex_buffer_object, cuda_cur_data (0), size,
                           cudaMemcpyDeviceToDevice, cuda_stream (0));
    check (ret, "cudaMemcpyAsync");

    cuda_release ();
  }
}

EXTERN void cuda_establish_bindings (void)
{
  int generic_version = 1;

  the_compute = (int_func_t)bind_it (kernel_name, "compute", variant_name, 0);
  if (the_compute == NULL) {
    if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
      the_compute = cuda_compute_2dimg;
    else
      the_compute = cuda_compute_3dmesh;

    PRINT_DEBUG ('h', "Using generic [%s] CUDA kernel launcher\n",
                 "cuda_compute");
  } else
    generic_version = 0;

  the_send_data =
      (void_func_t)bind_it (kernel_name, "send_data", variant_name, 0);

  // We need to gather a reference on the kernel only in case the generic
  // version of the launcher is used
  if (generic_version) {
    if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
      the_cuda_kernel_2dimg =
          (cuda_kernel_2dimg_t)bind_it (kernel_name, NULL, variant_name, 2);
    else
      the_cuda_kernel_3dmesh =
          (cuda_kernel_3dmesh_t)bind_it (kernel_name, NULL, variant_name, 2);
  }

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES && easypap_gl_buffer_sharing &&
      do_display) {
    the_cuda_update_texture = (cuda_update_texture_func_t)bind_it (
        kernel_name, "update_texture", variant_name, 0);
    if (the_cuda_update_texture == NULL) {
      printf ("specific update_texture not found\n");
      the_cuda_update_texture =
          (cuda_update_texture_func_t)cuda_update_texture_generic;
    }
  }
}
