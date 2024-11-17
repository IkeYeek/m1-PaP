
#include "easypap.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k scrollup -v seq
//
unsigned scrollup_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++) {
      int src = (i < DIM - 1) ? i + 1 : 0;
      for (int j = 0; j < DIM; j++)
        next_img (i, j) = cur_img (src, j);
    }

    swap_images ();
  }

  return 0;
}

// Tile inner computation
int scrollup_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++) {
    int src = (i < DIM - 1) ? i + 1 : 0;

    for (int j = x; j < x + width; j++)
      next_img (i, j) = cur_img (src, j);
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -l data/img/1024.png -k scrollup -v tiled
//
unsigned scrollup_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H);

    swap_images ();

  }

  return 0;
}

#ifdef ENABLE_OPENCL

//////////// OpenCL version using mask (ocl_ouf)
// Suggested cmdlines:
// ./run -l data/img/shibuya.png -k scrollup -g -v ocl_ouf
//

static cl_mem twin_buffer = 0, mask_buffer = 0;
static char *mask_file = NULL;

void scrollup_config_ocl_ouf (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a 1024x1024 mask
    if (DIM != 1024)
      exit_with_error ("scrollup-OpenCL-de-Ouf requires 1024x1024 images");
    mask_file = param;
  }
}

void scrollup_init_ocl_ouf (void)
{
  const int size = DIM * DIM * sizeof (unsigned);

  mask_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!mask_buffer)
    exit_with_error ("Failed to allocate mask buffer");

  twin_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!twin_buffer)
    exit_with_error ("Failed to allocate second input buffer");
}

void scrollup_draw_ocl_ouf (char *param)
{
  const int size = DIM * DIM * sizeof (unsigned);
  cl_int err;

  if (easypap_image_file == NULL)
    exit_with_error ("scrollup is prettier when applied to an image!");

  unsigned *tmp = malloc (size);
  if (mask_file) {
    int fd = open (mask_file, O_RDONLY);
    if (fd == -1)
      exit_with_error ("Cannot open file %s", mask_file);

    int n = read (fd, tmp, size);
    if (n != size)
      exit_with_error ("Cannot read from file %s", mask_file);
    close (fd);
  } else {
    // Draw a quick-n-dirty circle
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++) {
        const int mid = DIM / 2;
        int dist2     = (i - mid) * (i - mid) + (j - mid) * (j - mid);
        const int r1  = (DIM / 4) * (DIM / 4);
        const int r2  = (DIM / 2) * (DIM / 2);
        if (dist2 < r1)
          tmp[i * DIM + j] = ezv_a2c (255);
        else if (dist2 < r2)
          tmp[i * DIM + j] = ezv_a2c ((r2 - dist2) * 255 / (r2 - r1));
        else
          tmp[i * DIM + j] = ezv_a2c (0);
      }
  }
  // We send the mask buffer to GPU
  // (not need to send twin_buffer : its content will be erased during 1st
  // iteration)
  err = clEnqueueWriteBuffer (ocl_queue (0), mask_buffer, CL_TRUE, 0, size, tmp,
                              0, NULL, NULL);
  check (err, "Failed to write to extra buffer");

  free (tmp);

  img_data_replicate (); // Perform next_img = cur_img
}

unsigned scrollup_compute_ocl_ouf (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X,
                      GPU_SIZE_Y};     // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned color = ezv_rgb (255, 0, 0); // red
    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &twin_buffer);
    err |= clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem),
                           &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 3, sizeof (cl_mem),
                           &mask_buffer);
    err |=
        clSetKernelArg (ocl_compute_kernel (0), 4, sizeof (unsigned), &color);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Swap buffers
    {
      cl_mem tmp          = twin_buffer;
      twin_buffer         = ocl_next_buffer (0);
      ocl_next_buffer (0) = tmp;
    }
  }

  clFinish (ocl_queue (0));

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (0));

  return 0;
}

#endif
