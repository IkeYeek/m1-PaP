#include "ocl.h"
#include "arch_flags.h"
#include "constants.h"
#include "cpustat.h"
#include "debug.h"
#include "error.h"
#include "ezp_ctx.h"
#include "global.h"
#include "hooks.h"
#include "img_data.h"
#include "mesh_data.h"
#include "minmax.h"
#include "time_macros.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <SDL2/SDL_opengl.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenGL/CGLContext.h>
#include <OpenGL/CGLCurrent.h>
#else
#include <CL/opencl.h>
#include <GL/glx.h>
#endif

#define _stringify(s) #s
#define stringify(s) _stringify (s)

#define MESH_NEIGHBOR_ROUND 64U

#define MAX_PLATFORMS 3
#define MAX_DEVICES 5
#define MAX_KERNELS 32

unsigned GPU_SIZE   = 0;
unsigned TILE       = 0;
unsigned GPU_SIZE_X = 0;
unsigned GPU_SIZE_Y = 0;

static size_t max_workgroup_size = 0;

static cl_platform_id chosen_platform = NULL;
cl_device_id chosen_device            = NULL;
cl_program program; // compute program

cl_context context;
cl_kernel update_kernel;
static cl_kernel bench_kernel; // bench null kernel

cl_mem tex_buffer;
cl_mem neighbor_soa_buffer;

ocl_gpu_t ocl_gpu[MAX_DEVICES];
unsigned ocl_nb_gpus = 0;

static size_t file_size (const char *filename)
{
  struct stat sb;

  if (stat (filename, &sb) < 0)
    exit_with_error ("Cannot access \"%s\" kernel file (%s)", filename,
                     strerror (errno));

  return sb.st_size;
}

static char *file_load (const char *filename, const char *common)
{
  FILE *f, *fc;
  char *b;
  size_t s, sc = 0;
  size_t r;

  s = file_size (filename);
  if (common != NULL)
    sc = file_size (common);
  b = malloc (s + sc + 2);
  if (!b)
    exit_with_error ("Malloc failed (%s)", strerror (errno));

  if (common != NULL) {
    fc = fopen (common, "r");
    if (fc == NULL)
      exit_with_error ("Cannot open \"%s\" file (%s)", common,
                       strerror (errno));

    r = fread (b, sc, 1, fc);
    if (r != 1)
      exit_with_error ("fread failed (%s)", strerror (errno));
  }

  f = fopen (filename, "r");
  if (f == NULL)
    exit_with_error ("Cannot open \"%s\" file (%s)", filename,
                     strerror (errno));

  b[sc] = '\n';
  r     = fread (b + sc + 1, s, 1, f);
  if (r != 1)
    exit_with_error ("fread failed (%s)", strerror (errno));

  b[s + sc + 1] = '\0';

  return b;
}

unsigned easypap_number_of_gpus_ocl (void)
{
  return ocl_nb_gpus;
}

void ocl_acquire (void)
{
  if (do_display && easypap_gl_buffer_sharing) {
    cl_int err;

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
      err = clEnqueueAcquireGLObjects (queue, 1, &tex_buffer, 0, NULL, NULL);
    } else {
      err = clEnqueueAcquireGLObjects (queue, 1, &cur_buffer, 0, NULL, NULL);
      err |= clEnqueueAcquireGLObjects (queue, 1, &next_buffer, 0, NULL, NULL);
    }

    check (err, "Failed to acquire lock");
  }
}

void ocl_release (void)
{
  if (do_display && easypap_gl_buffer_sharing) {
    cl_int err;

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
      err = clEnqueueReleaseGLObjects (queue, 1, &tex_buffer, 0, NULL, NULL);
    } else {
      err = clEnqueueReleaseGLObjects (queue, 1, &cur_buffer, 0, NULL, NULL);
      err |= clEnqueueReleaseGLObjects (queue, 1, &next_buffer, 0, NULL, NULL);
    }

    check (err, "Failed to release lock");
  }
}

static void ocl_show_config (int quit, int verbose)
{
  cl_platform_id pf[MAX_PLATFORMS];
  cl_int nbp      = 0;
  cl_int chosen_p = -1, chosen_d = -1;
  char *glRenderer = NULL;
  char *str        = NULL;
  cl_int err;

  if (do_display)
    glRenderer = (char *)glGetString (GL_RENDERER);

  // Get list of platforms
  err = clGetPlatformIDs (MAX_PLATFORMS, pf, (cl_uint *)&nbp);
  check (err, "Failed to get platform IDs");

  if (verbose == 2)
    printf ("%d OpenCL platforms detected\n", nbp);

  str = getenv ("PLATFORM");
  if (str != NULL)
    chosen_p = atoi (str);

  if (!quit && chosen_p >= nbp)
    exit_with_error (
        "Requested platform number (%d) should be in [0..%d] range", chosen_p,
        nbp - 1);

  str = getenv ("DEVICE");
  if (str != NULL)
    chosen_d = atoi (str);

  if (chosen_p == -1 && chosen_d != -1)
    chosen_p = 0;

  // Go through the list of platforms
  for (cl_uint p = 0; p < nbp; p++) {
    char name[1024], vendor[1024];
    cl_device_id devices[MAX_DEVICES];
    cl_int nbd = 0;
    cl_device_type dtype;

    if (chosen_p == p)
      chosen_platform = pf[p];

    err = clGetPlatformInfo (pf[p], CL_PLATFORM_NAME, 1024, name, NULL);
    check (err, "Failed to get Platform Info");

    err = clGetPlatformInfo (pf[p], CL_PLATFORM_VENDOR, 1024, vendor, NULL);
    check (err, "Failed to get Platform Info");

    if (verbose == 2)
      printf ("Platform %d: %s (%s)\n", p, name, vendor);

    err = clGetDeviceIDs (pf[p], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices,
                          (cl_uint *)&nbd);

    if (!quit && chosen_p == p && chosen_d >= nbd)
      exit_with_error (
          "Requested device number (%d) should be in [0..%d] range", chosen_d,
          nbd - 1);

    // The chosen platform provides only one device, so we take device[0]
    if (chosen_p == p && chosen_d == -1 && nbd == 1) {
      chosen_d      = 0;
      chosen_device = devices[0];

      ocl_gpu[ocl_nb_gpus++].device = chosen_device;
    }

    // Go through the list of devices for platform p
    for (cl_uint d = 0; d < nbd; d++) {
      int disp = 0;

      err = clGetDeviceInfo (devices[d], CL_DEVICE_NAME, 1024, name, NULL);
      check (err, "Cannot get type of device");

      err = clGetDeviceInfo (devices[d], CL_DEVICE_TYPE,
                             sizeof (cl_device_type), &dtype, NULL);
      check (err, "Cannot get type of device");

      // If user specified no PLATFORM/DEVICE, just take the first GPU found
      if (dtype == CL_DEVICE_TYPE_GPU && (chosen_p == -1 || chosen_p == p) &&
          (chosen_d == -1)) {
        chosen_p                      = p;
        chosen_platform               = pf[p];
        chosen_d                      = d;
        chosen_device                 = devices[d];
        disp                          = 1;
        ocl_gpu[ocl_nb_gpus++].device = chosen_device;
      } else if (chosen_p == p) {
        if (chosen_d == d) {
          chosen_device                 = devices[d];
          disp                          = 1;
          ocl_gpu[ocl_nb_gpus++].device = chosen_device;
        } else if (chosen_d == -1 && d == nbd - 1) {
          // Last chance to select device
          chosen_d                      = 0;
          chosen_device                 = devices[0];
          disp                          = 1;
          ocl_gpu[ocl_nb_gpus++].device = chosen_device;
        } else if (dtype == CL_DEVICE_TYPE_GPU && use_multiple_gpu) {
          disp                          = 1;
          ocl_gpu[ocl_nb_gpus++].device = devices[d];
        }
      } else if (dtype == CL_DEVICE_TYPE_GPU && use_multiple_gpu) {
        disp                          = 1;
        ocl_gpu[ocl_nb_gpus++].device = devices[d];
      }

      if (verbose == 2)
        printf ("%s Device %d: %s [%s]\n", disp ? "+++" : "---", d,
                (dtype == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", name);
      else if (verbose == 1 && disp)
        printf ("Using OpenCL Device: %s [%s]\n",
                (dtype == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", name);
    }
  }

  if (verbose == 2)
    printf ("    => %d device(s) used\n", ocl_nb_gpus);

  if (verbose == 2 && glRenderer != NULL)
    printf ("Note: OpenGL renderer uses [%s]\n", glRenderer);

  if (quit)
    exit (0);
}

void ocl_init (int show_config, int silent)
{
  cl_int err;
  int verbose = 0;

  if (!silent) {
    if (show_config || debug_enabled ('o'))
      verbose = 2;
    else
      verbose = 1;
  }

  ocl_show_config (show_config, verbose);

  if (chosen_device == NULL)
    exit_with_error ("Device could not be automatically chosen: please use "
                     "PLATFORM and DEVICE to specify target");

  err = clGetDeviceInfo (chosen_device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof (size_t), &max_workgroup_size, NULL);
  check (err, "Cannot get max workgroup size");

  if (do_display && easypap_gl_buffer_sharing) {
    ezv_switch_to_context (ctx[0]);
#ifdef __APPLE__
    CGLContextObj cgl_context          = CGLGetCurrentContext ();
    CGLShareGroupObj sharegroup        = CGLGetShareGroup (cgl_context);
    cl_context_properties properties[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)sharegroup, 0};
#else
    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR,
        (cl_context_properties)glXGetCurrentContext (),
        CL_GLX_DISPLAY_KHR,
        (cl_context_properties)glXGetCurrentDisplay (),
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)chosen_platform,
        0};
    if (easypap_mode == EASYPAP_MODE_3D_MESHES)
      properties[1] = (cl_context_properties)ezv_glcontext (ctx[0]);
#endif

    context = clCreateContext (properties, 1, &chosen_device, NULL, NULL, &err);
  } else {
    cl_device_id devices[MAX_DEVICES];
    for (int g = 0; g < ocl_nb_gpus; g++)
      devices[g] = ocl_gpu[g].device;
    context = clCreateContext (NULL, ocl_nb_gpus, devices, NULL, NULL, &err);
  }

  check (err, "Failed to create compute context. Please make sure OpenCL and "
              "OpenGL both use the same device (--show-devices).");

  // Create command queues
  //
  for (int g = 0; g < ocl_nb_gpus; g++) {
    ocl_gpu[g].q = clCreateCommandQueue (context, ocl_gpu[g].device,
                                         CL_QUEUE_PROFILING_ENABLE, &err);
    check (err, "Failed to create command queue.\nPlease make sure both OpenCL and "
                "OpenGL use the same device (./run --show-devices)\n"
                "or use --no-gl-buffer-share (-nbs) option.");
    // printf ("queue %p for device %p\n", ocl_gpu[g].queue, ocl_gpu[g].device);
  }
}

void ocl_alloc_buffers (void)
{
  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    const unsigned size = DIM * DIM * sizeof (unsigned);

    // Allocate buffers inside device memory
    //
    for (int g = 0; g < ocl_nb_gpus; g++) {
      ocl_gpu[g].curb =
          clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
      if (!ocl_gpu[g].curb)
        exit_with_error ("Failed to allocate input buffer");

      ocl_gpu[g].nextb =
          clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
      if (!ocl_gpu[g].nextb)
        exit_with_error ("Failed to allocate output buffer");
    }
    // Shared texture
    if (do_display && easypap_gl_buffer_sharing) {
      int gl_buffer_ids[1];

      ezv_get_shareable_buffer_ids (ctx[0], gl_buffer_ids);

      cl_int err;
      // Shared texture buffer with OpenGL
      //
      tex_buffer = clCreateFromGLTexture (context, CL_MEM_READ_WRITE,
                                          GL_TEXTURE_2D, 0, gl_buffer_ids[0], &err);
      check (err, "Failed to map texture buffer\n");

      PRINT_DEBUG ('o', "OpenGL buffers shared with OpenCL\n");
    }
  } else { // 3D_MESHES
    cl_int err;
    if (do_display && easypap_gl_buffer_sharing) {
      int gl_buffer_ids[2];

      ezv_get_shareable_buffer_ids (ctx[0], gl_buffer_ids);

      cur_buffer = clCreateFromGLBuffer (context, CL_MEM_READ_WRITE,
                                         gl_buffer_ids[0], &err);
      check (err, "Failed to map value buffer #0\n");

      next_buffer = clCreateFromGLBuffer (context, CL_MEM_READ_WRITE,
                                          gl_buffer_ids[1], &err);
      check (err, "Failed to map value buffer #1\n");

      PRINT_DEBUG ('o', "OpenGL buffers shared with OpenCL\n");
    } else {
      const unsigned size = NB_CELLS * sizeof (float);
      // Allocate buffers inside device memory
      //
      for (int g = 0; g < ocl_nb_gpus; g++) {
        ocl_gpu[g].curb =
            clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
        if (!ocl_gpu[g].curb)
          exit_with_error ("Failed to allocate value buffer #0");

        ocl_gpu[g].nextb =
            clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
        if (!ocl_gpu[g].nextb)
          exit_with_error ("Failed to allocate value buffer #1");
      }
    }

    if (ocl_nb_gpus == 1) {
      // Buffers hosting neighbors
      mesh_data_build_neighbors_soa (TILE); // GPU_SIZE is rounded accordingly

      const unsigned size =
          neighbor_soa_offset * easypap_mesh_desc.max_neighbors * sizeof (int);

      neighbor_soa_buffer =
          clCreateBuffer (context, CL_MEM_READ_ONLY, size, NULL, NULL);
      if (!neighbor_soa_buffer)
        exit_with_error ("Failed to allocate neighbor buffer\n");

      for (int g = 0; g < ocl_nb_gpus; g++) {
        err = clEnqueueWriteBuffer (ocl_gpu[g].q, neighbor_soa_buffer, CL_TRUE,
                                    0, size, neighbors_soa, 0, NULL, NULL);
        check (err, "Failed to write to neighbor_soa_buffer");
      }
    }
  }
}

static void ocl_list_variants (void)
{
  cl_kernel kernels[MAX_KERNELS];
  char buffer[1024];
  cl_uint kernels_found = 0;
  cl_int err;

  err =
      clCreateKernelsInProgram (program, MAX_KERNELS, kernels, &kernels_found);
  check (err, "Failed to get list of kernels from program\n");

  for (int k = 0; k < kernels_found; k++) {
    size_t len;
    err = clGetKernelInfo (kernels[k], CL_KERNEL_FUNCTION_NAME, 1024, buffer,
                           &len);
    check (err, "Failed to get name of kernel\n");

    printf ("%s\n", buffer);
  }

  exit (EXIT_SUCCESS);
}

#define CALIBRATION_BURST 1
#define CALIBRATION_ITER 64

static void calibrate (void)
{
  size_t global[1] = {4096 * 64}; // global domain size for our calculation
  size_t local[1]  = {16};        // local domain size for our calculation
  cl_event events[CALIBRATION_BURST];
  int64_t t;
  cl_int err;

  bench_kernel = clCreateKernel (program, "bench_kernel", &err);
  check (err, "Failed to create bench kernel");

  for (int g = 0; g < ocl_nb_gpus; g++) {
    // Warmup
    for (unsigned it = 0; it < 2; it++) {

      err = clEnqueueNDRangeKernel (ocl_gpu[g].q, bench_kernel, 1, NULL, global,
                                    local, 0, NULL, NULL);
      check (err, "Failed to execute bench kernel");
    }
    clFinish (ocl_gpu[g].q);

    for (int i = 0; i < CALIBRATION_ITER; i++) {
      for (unsigned it = 0; it < CALIBRATION_BURST; it++)
        err = clEnqueueNDRangeKernel (ocl_gpu[g].q, bench_kernel, 1, NULL,
                                      global, local, 0, NULL, &events[it]);

      clFinish (ocl_gpu[g].q);
      t = what_time_is_it ();

      cl_ulong end;

      clGetEventProfilingInfo (events[CALIBRATION_BURST - 1],
                               CL_PROFILING_COMMAND_END, sizeof (cl_ulong),
                               &end, NULL);

      // printf ("Calibration(gpu %d): getclock= %" PRId64 ", clGetEvent= %"
      // PRId64
      // "\n", g, t, end / 1000);

      if (i == 0) {
        ocl_gpu[g].calibration_delta = t - (end / 1000);
      } else {
        ocl_gpu[g].calibration_delta =
            MAX (ocl_gpu[g].calibration_delta, t - (end / 1000));
      }

      for (unsigned i = 0; i < CALIBRATION_BURST; i++)
        clReleaseEvent (events[i]);
    }
    // printf ("Calibration value(gpu %d) = %" PRId64 "\n", g,
    // ocl_gpu[g].calibration_delta);
  }
}

void ocl_build_program (int list_variants)
{
  cl_int err;
  char *str = NULL;
  char buffer[1024];

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
    if (TILE_W * TILE_H > max_workgroup_size)
      exit_with_error ("TILE_W (%d) x TILE_H (%d) cannot exceed "
                       "CL_DEVICE_MAX_WORK_GROUP_SIZE (%ld)",
                       TILE_W, TILE_H, max_workgroup_size);
  } else {
    str = getenv ("TILE");
    if (str != NULL) {
      TILE = atoi (str);
      if (TILE % 32 != 0)
        exit_with_error ("Workgroup size (TILE) should be a multiple of 32");
    } else
      TILE = MESH_NEIGHBOR_ROUND;
    GPU_SIZE = ROUND_TO_MULTIPLE (NB_CELLS, TILE);
  }

  // Load program source into memory
  //
  sprintf (buffer, "kernel/ocl/%s.cl", kernel_name);
  const char *opencl_prog = file_load (buffer, NULL);

  // Attach program source to context
  //
  program = clCreateProgramWithSource (context, 1, &opencl_prog, NULL, &err);
  check (err, "Failed to create program");

  // Compile program
  //
  char *debug_str = "";
  if (debug_enabled ('o'))
    debug_str = "-DDEBUG=1";

  char *endianness = "";
  if (IS_LITTLE_ENDIAN)
    endianness = "-DIS_LITTLE_ENDIAN";
  else
    endianness = "-DIS_BIG_ENDIAN";

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    if (config_param)
      sprintf (buffer,
               " -cl-strict-aliasing -cl-fast-relaxed-math"
               " -cl-mad-enable"
               " -DDIM=%d -DGPU_SIZE_X=%d -DGPU_SIZE_Y=%d -DTILE_W=%d"
               " -DTILE_H=%d -DKERNEL_%s"
               " -DPARAM=%s %s %s -D%s",
               DIM, GPU_SIZE_X, GPU_SIZE_Y, TILE_W, TILE_H, kernel_name,
               config_param, debug_str, endianness, stringify (ARCH));
    else
      sprintf (buffer,
               " -cl-strict-aliasing -cl-fast-relaxed-math"
               " -cl-mad-enable"
               " -DDIM=%d -DGPU_SIZE_X=%d -DGPU_SIZE_Y=%d -DTILE_W=%d"
               " -DTILE_H=%d -DKERNEL_%s %s %s -D%s",
               DIM, GPU_SIZE_X, GPU_SIZE_Y, TILE_W, TILE_H, kernel_name,
               debug_str, endianness, stringify (ARCH));
  } else {
    sprintf (buffer,
             " -cl-strict-aliasing -cl-fast-relaxed-math"
             " -cl-mad-enable"
             " -DNB_CELLS=%d -DGPU_SIZE=%d -DTILE=%d -DMAX_NEIGHBORS=%d"
             " -DKERNEL_%s %s %s -D%s",
             NB_CELLS, GPU_SIZE, TILE, easypap_mesh_desc.max_neighbors,
             kernel_name, debug_str, endianness, stringify (ARCH));
  }
  // printf ("[OpenCL flags: %s]\n", buffer);

  err = clBuildProgram (program, 0, NULL, buffer, NULL, NULL);

  // Display compiler log
  //
  {
    size_t len;

    clGetProgramBuildInfo (program, chosen_device, CL_PROGRAM_BUILD_LOG, 0,
                           NULL, &len);

    if (len > 2 && len <= 2048) {
      char buffer[len];

      fprintf (stderr, "--- OpenCL Compiler log ---\n");
      clGetProgramBuildInfo (program, chosen_device, CL_PROGRAM_BUILD_LOG,
                             sizeof (buffer), buffer, NULL);
      fprintf (stderr, "%s\n", buffer);
      fprintf (stderr, "---------------------------\n");
    }
  }

  if (err != CL_SUCCESS)
    exit_with_error ("Failed to build program");

  if (list_variants)
    ocl_list_variants ();

  // Create the compute kernels in the program we wish to run
  //
  sprintf (buffer, "%s_%s", kernel_name, variant_name);
  for (int g = 0; g < ocl_nb_gpus; g++) {
    ocl_gpu[g].kernel = clCreateKernel (program, buffer, &err);
    check (err, "Failed to create compute kernel <%s>", buffer);
  }

  PRINT_DEBUG ('o', "Using OpenCL kernel: %s_%s\n", kernel_name, variant_name);

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES && easypap_gl_buffer_sharing) {
    // First look for kernel-specific version of update_texture
    sprintf (buffer, "%s_update_texture", kernel_name);
    update_kernel = clCreateKernel (program, buffer, &err);
    if (err != CL_SUCCESS) {
      // Fall back to generic version
      update_kernel = clCreateKernel (program, "update_texture", &err);
      check (err, "Failed to create kernel <update_texture>");
    }
  }

  calibrate ();

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
    printf ("Using %dx%d workitems grouped in %dx%d tiles\n", GPU_SIZE_X,
            GPU_SIZE_Y, TILE_W, TILE_H);
  else
    printf ("Using %d workitems grouped in %d tiles\n", GPU_SIZE, TILE);
}

void ocl_send_data (void)
{
  if (the_send_data != NULL) {
    the_send_data ();
    PRINT_DEBUG ('i', "Init phase 7 : Initial data transferred to OpenCL "
                      "device (user-defined callback)\n");
  } else if (ocl_nb_gpus == 1) {
    cl_int err;
    const int g = 0;

    if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {

      const unsigned size = DIM * DIM * sizeof (unsigned);

      err = clEnqueueWriteBuffer (ocl_gpu[g].q, cur_buffer, CL_TRUE, 0, size,
                                  image, 0, NULL, NULL);
      check (err, "Failed to write to cur_buffer");

      err = clEnqueueWriteBuffer (ocl_gpu[g].q, next_buffer, CL_TRUE, 0, size,
                                  alt_image, 0, NULL, NULL);
      check (err, "Failed to write to next_buffer");

      PRINT_DEBUG (
          'i', "Init phase 7 : Initial data transferred to OpenCL device\n");

    } else { // 3D_MESHES

      const unsigned size = NB_CELLS * sizeof (float);
      const int g         = 0;

      ocl_acquire ();

      err = clEnqueueWriteBuffer (ocl_gpu[g].q, ocl_gpu[g].curb, CL_TRUE, 0,
                                  size, mesh_data, 0, NULL, NULL);
      check (err, "Failed to write to cur_buffer");

      err = clEnqueueWriteBuffer (ocl_gpu[g].q, ocl_gpu[g].nextb, CL_TRUE, 0,
                                  size, alt_mesh_data, 0, NULL, NULL);
      check (err, "Failed to write to next_buffer");

      ocl_release ();

      PRINT_DEBUG (
          'i', "Init phase 7 : Initial data transferred to OpenCL device\n");
    }
  }
}

void ocl_retrieve_data (void)
{
  cl_int err;

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {

    const unsigned size = DIM * DIM * sizeof (unsigned);

    err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0, size, image, 0,
                               NULL, NULL);
    check (err, "Failed to read from cur_buffer");
  } else {
    const unsigned size = NB_CELLS * sizeof (float);

    ocl_acquire ();

    err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0, size, mesh_data,
                               0, NULL, NULL);
    check (err, "Failed to read from cur_buffer");

    ocl_release ();
  }

  // PRINT_DEBUG ('o', "Data retrieved from OpenCL device\n");
}

static unsigned ocl_compute_2dimg (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X,
                      GPU_SIZE_Y};     // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  uint64_t clock =
      monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE, 0));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp  = cur_buffer;
      cur_buffer  = next_buffer;
      next_buffer = tmp;
    }
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM,
                       easypap_gpu_lane (TASK_TYPE_COMPUTE, 0));

  return 0;
}

static unsigned ocl_compute_3dmesh (unsigned nb_iter)
{
  size_t global[1] = {GPU_SIZE}; // global domain size for our calculation
  size_t local[1]  = {TILE};     // local domain size for our calculation
  cl_int err;

  ocl_acquire ();

  uint64_t clock =
      monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE, 0));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (cl_mem),
                           &neighbor_soa_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 1, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp  = cur_buffer;
      cur_buffer  = next_buffer;
      next_buffer = tmp;
      if (do_display)
        ezv_switch_color_buffers (ctx[0]);
    }
  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, NB_CELLS, 0,
                       easypap_gpu_lane (TASK_TYPE_COMPUTE, 0));

  ocl_release ();

  return 0;
}

unsigned ocl_compute (unsigned nb_iter)
{
  if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
    return ocl_compute_2dimg (nb_iter);
  else
    return ocl_compute_3dmesh (nb_iter);
}

void ocl_establish_bindings (void)
{
  the_compute = bind_it (kernel_name, "compute", variant_name, 0);
  if (the_compute == NULL) {
    the_compute = ocl_compute;
    PRINT_DEBUG ('c', "Using generic [%s] OpenCL kernel launcher\n",
                 "ocl_compute");
  }
  the_send_data = bind_it (kernel_name, "send_data", variant_name, 0);
}

void ocl_update_texture (void)
{
  size_t global[2] = {DIM, DIM}; // global domain size for our calculation
  size_t local[2]  = {16, 16};   // local domain size for our calculation
  cl_int err;

  ocl_acquire ();

  // Set kernel arguments
  //
  err = 0;
  err |= clSetKernelArg (update_kernel, 0, sizeof (cl_mem), &cur_buffer);
  err |= clSetKernelArg (update_kernel, 1, sizeof (cl_mem), &tex_buffer);
  check (err, "Failed to set kernel arguments");

  err = clEnqueueNDRangeKernel (queue, update_kernel, 2, NULL, global, local, 0,
                                NULL, NULL);
  check (err, "Failed to execute update_texture kernel");

  ocl_release ();

  clFinish (queue);
}

size_t ocl_get_max_workgroup_size (void)
{
  return max_workgroup_size;
}

static inline int64_t ocl_start_time (cl_event evt, unsigned gpu_no)
{
  cl_ulong t_start;

  clGetEventProfilingInfo (evt, CL_PROFILING_COMMAND_START, sizeof (cl_ulong),
                           &t_start, NULL);

  return (t_start / 1000) + ocl_gpu[gpu_no].calibration_delta;
}

static inline int64_t ocl_end_time (cl_event evt, unsigned gpu_no)
{
  cl_ulong t_end;

  clGetEventProfilingInfo (evt, CL_PROFILING_COMMAND_END, sizeof (cl_ulong),
                           &t_end, NULL);

  return (t_end / 1000) + ocl_gpu[gpu_no].calibration_delta;
}

int64_t ocl_monitor (cl_event evt, int x, int y, int width, int height,
                     task_type_t task_type, unsigned gpu_no)
{
  int64_t start, end;
  unsigned gpu_lane = easypap_gpu_lane (task_type, gpu_no);

  start = ocl_start_time (evt, gpu_no);
  end   = ocl_end_time (evt, gpu_no);

  int64_t now = what_time_is_it ();
  if (end > now)
    PRINT_DEBUG ('c',
                 "Warning: end of kernel (%s) ahead of current time by %" PRId64
                 " µs\n",
                 task_type == TASK_TYPE_COMPUTE ? "TASK_TYPE_COMPUTE"
                                                : "TASK_TYPE_TRANSFER",
                 end - now);

  // PRINT_DEBUG ('o', "[%s] start: %" PRId64 ", end: %" PRId64 "\n", "kernel",
  //              start, end);

  monitoring_gpu_tile (x, y, width, height, gpu_lane, start, end, task_type);

  return end - start;
}

static void callback (cl_event event, cl_int event_command_status,
                      void *user_data)
{
  ocl_stamp_t *stamp = (ocl_stamp_t *)user_data;

  if (event_command_status == CL_COMPLETE) {
    stamp->end = what_time_is_it ();
    printf ("Event time stamp : %" PRId64 " to %" PRId64 "\n", stamp->start,
            stamp->end);
  }
}

void ocl_link_stamp (cl_event evt, ocl_stamp_t *stamp)
{
  cl_int err;

  err = clSetEventCallback (evt, CL_COMPLETE, callback, stamp);
  check (err, "Failed to set event callback");
}

const char *ocl_GetError (cl_int error)
{
  switch (error) {
  // run-time and JIT compiler errors
  case 0:
    return "CL_SUCCESS";
  case -1:
    return "CL_DEVICE_NOT_FOUND";
  case -2:
    return "CL_DEVICE_NOT_AVAILABLE";
  case -3:
    return "CL_COMPILER_NOT_AVAILABLE";
  case -4:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5:
    return "CL_OUT_OF_RESOURCES";
  case -6:
    return "CL_OUT_OF_HOST_MEMORY";
  case -7:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8:
    return "CL_MEM_COPY_OVERLAP";
  case -9:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case -10:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11:
    return "CL_BUILD_PROGRAM_FAILURE";
  case -12:
    return "CL_MAP_FAILURE";
  case -13:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case -16:
    return "CL_LINKER_NOT_AVAILABLE";
  case -17:
    return "CL_LINK_PROGRAM_FAILURE";
  case -18:
    return "CL_DEVICE_PARTITION_FAILED";
  case -19:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

  // compile-time errors
  case -30:
    return "CL_INVALID_VALUE";
  case -31:
    return "CL_INVALID_DEVICE_TYPE";
  case -32:
    return "CL_INVALID_PLATFORM";
  case -33:
    return "CL_INVALID_DEVICE";
  case -34:
    return "CL_INVALID_CONTEXT";
  case -35:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case -36:
    return "CL_INVALID_COMMAND_QUEUE";
  case -37:
    return "CL_INVALID_HOST_PTR";
  case -38:
    return "CL_INVALID_MEM_OBJECT";
  case -39:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40:
    return "CL_INVALID_IMAGE_SIZE";
  case -41:
    return "CL_INVALID_SAMPLER";
  case -42:
    return "CL_INVALID_BINARY";
  case -43:
    return "CL_INVALID_BUILD_OPTIONS";
  case -44:
    return "CL_INVALID_PROGRAM";
  case -45:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46:
    return "CL_INVALID_KERNEL_NAME";
  case -47:
    return "CL_INVALID_KERNEL_DEFINITION";
  case -48:
    return "CL_INVALID_KERNEL";
  case -49:
    return "CL_INVALID_ARG_INDEX";
  case -50:
    return "CL_INVALID_ARG_VALUE";
  case -51:
    return "CL_INVALID_ARG_SIZE";
  case -52:
    return "CL_INVALID_KERNEL_ARGS";
  case -53:
    return "CL_INVALID_WORK_DIMENSION";
  case -54:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case -55:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case -56:
    return "CL_INVALID_GLOBAL_OFFSET";
  case -57:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case -58:
    return "CL_INVALID_EVENT";
  case -59:
    return "CL_INVALID_OPERATION";
  case -60:
    return "CL_INVALID_GL_OBJECT";
  case -61:
    return "CL_INVALID_BUFFER_SIZE";
  case -62:
    return "CL_INVALID_MIP_LEVEL";
  case -63:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64:
    return "CL_INVALID_PROPERTY";
  case -65:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66:
    return "CL_INVALID_COMPILER_OPTIONS";
  case -67:
    return "CL_INVALID_LINKER_OPTIONS";
  case -68:
    return "CL_INVALID_DEVICE_PARTITION_COUNT";

  // extension errors
  case -1000:
    return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001:
    return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002:
    return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003:
    return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004:
    return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005:
    return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default:
    return "Unknown OpenCL error";
  }
}
