#include "easypap.h"
#include "ezm_time.h"
#include "rle_lexer.h"

#include <CL/cl.h>
#include <numa.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define LIFE_COLOR (ezv_rgb (255, 255, 0))

typedef char cell_t;

static cell_t *restrict __attribute__ ((aligned (64))) _table           = NULL;
static cell_t *restrict __attribute__ ((aligned (64))) _alternate_table = NULL;

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

void life_omp_ocl_init (void)
{
  // life_omp_ocl_init may be (indirectly) called several times so we check if
  // data were already allocated
  if (_table == NULL) {
    unsigned size = DIM * DIM * sizeof (cell_t);

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d ", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
}
#define ENABLE_OPENCL
#ifdef ENABLE_OPENCL

#ifndef TASKV
#define BORDER_SIZE 10
#define GPU_CPU_SYNC_FREQ BORDER_SIZE // a little more explicit when used...

static ezp_gpu_event_footprint_t kernel_fp[2];
static uint64_t kernel_durations[2];
static unsigned true_iter_number;

/* === kernel/compute functions === */
static inline void enqueue_kernel (cl_int err, size_t global[2],
                                   size_t local[2], uint64_t *clock)
{
  err = 0;
  err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                         &ocl_cur_buffer (0));
  err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                         &ocl_next_buffer (0));
  check (err, "Error setting kernel arguments");

  *clock = ezm_gettime ();
  err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2, NULL,
                                global, local, 0, NULL,
                                ezp_ocl_eventptr (EVENT_START_KERNEL, 0));
  check (err, "Error enqueuing kernel");
  clFlush (ocl_queue (0));
}

static inline void compute_cpu (unsigned *change)
{

  int border_tiles = (BORDER_SIZE * 2) / TILE_H + 1;
  int cpu_start_y  = kernel_fp[0].h - (border_tiles * TILE_H);
#pragma omp parallel for collapse(2) schedule(runtime)
  for (int y = cpu_start_y; y < DIM; y += TILE_H) {
    for (int x = kernel_fp[1].x; x < DIM; x += TILE_W) {
      *change |= do_tile (x, y, TILE_W, TILE_H);
    }
  }
}

static inline void finish_and_time (uint64_t clock)
{
  kernel_durations[1] = ezm_gettime () - clock;
  clFinish (ocl_queue (0));
  kernel_durations[0] = ezp_gpu_event_monitor (
      0, EVENT_START_KERNEL, clock, &kernel_fp[0], TASK_TYPE_COMPUTE, 0);
  ezp_gpu_event_reset ();
}

static inline ocl_swap_tables ()
{
  cl_mem tmp          = ocl_cur_buffer (0);
  ocl_cur_buffer (0)  = ocl_next_buffer (0);
  ocl_next_buffer (0) = tmp;
  cell_t *tmp2        = _table;
  _table              = _alternate_table;
  _alternate_table    = tmp2;
}

static inline ocl_sync_borders (cl_int err)
{
  unsigned true_gpu_size =
      sizeof (cell_t) * DIM * (kernel_fp[0].h - BORDER_SIZE);

  err = clEnqueueReadBuffer (
      ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
      sizeof (cell_t) * DIM * (kernel_fp[0].h - BORDER_SIZE * 2),
      sizeof (cell_t) * DIM * BORDER_SIZE,
      _table + DIM * (kernel_fp[0].h - BORDER_SIZE * 2), 0, NULL, NULL);
  check (err, "Err syncing host to device");

  size_t border_offset_elements = DIM * (kernel_fp[0].h - BORDER_SIZE);

  err =
      clEnqueueWriteBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
                            true_gpu_size, BORDER_SIZE * DIM * sizeof (cell_t),
                            _table + border_offset_elements, 0, NULL, NULL);
  check (err, "Err syncing device to host");
}

/* === initializations === */

void life_omp_ocl_config_ocl (char *params)
{
  easypap_gl_buffer_sharing = 0;
  life_omp_ocl_config (params);
}

void life_omp_ocl_config_ocl_adaptive (char *params)
{
  life_omp_ocl_config_ocl (params);
}

void life_omp_ocl_config_ocl_mt (char *params)
{
  life_omp_ocl_config_ocl (params);
}

void life_omp_ocl_init_ocl ()
{
  kernel_fp[0].x = 0;
  kernel_fp[0].y = 0;
  kernel_fp[0].w = DIM;
  kernel_fp[0].h = (NB_TILES_Y / 2) * TILE_H;

  kernel_fp[1].x = 0;
  kernel_fp[1].y = kernel_fp[0].y + kernel_fp[0].h;
  kernel_fp[1].w = DIM;
  kernel_fp[1].h = DIM - kernel_fp[1].y;

  kernel_durations[0] = 0;
  kernel_durations[1] = 0;

  true_iter_number = 0;
  life_omp_ocl_init ();
  life_omp_ocl_ft ();
}

void life_omp_ocl_init_ocl_adaptive ()
{
  life_omp_ocl_init_ocl ();
}

void life_omp_ocl_init_ocl_mt ()
{
  life_omp_ocl_init_ocl ();
}

/* === computes === */

unsigned life_omp_ocl_compute_ocl (unsigned nb_iter)
{
  size_t global[2] = {DIM,
                      kernel_fp[0].h}; // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;
  uint64_t clock;
  unsigned change = 0;

  for (unsigned iter = 1; iter <= nb_iter; iter++) {
    enqueue_kernel (err, global, local, &clock);
    compute_cpu (&change);
    finish_and_time (clock);
    ocl_swap_tables ();
    if (++true_iter_number % GPU_CPU_SYNC_FREQ == 0 && true_iter_number > 0)
      ocl_sync_borders (err);
  }
  return 0;
}
unsigned life_omp_ocl_compute_ocl_mt (unsigned nb_iter)
{
  size_t global[2] = {DIM,
                      kernel_fp[0].h}; // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;
  uint64_t clock;
  unsigned change  = 0;
  int border_tiles = (BORDER_SIZE * 2) / TILE_H + 1;
  int cpu_start_y  = kernel_fp[0].h - (border_tiles * TILE_H);

#pragma omp parallel master
  for (unsigned iter = 1; iter <= nb_iter; iter++) {
#pragma omp task
    enqueue_kernel (err, global, local, &clock);
#pragma omp taskloop collapse(2)
    for (int y = cpu_start_y; y < DIM; y += TILE_H) {
      for (int x = 0; x < DIM; x += TILE_W) {
        change |= do_tile (x, y, TILE_W, TILE_H);
      }
    }
    finish_and_time (clock);
    ocl_swap_tables ();
    if (++true_iter_number % GPU_CPU_SYNC_FREQ == 0 && true_iter_number > 0)
      ocl_sync_borders (err);
  }
  return 0;
}
static inline bool much_greater_than (uint64_t a, uint64_t b)
{
  return a > b * 10;
}

unsigned life_omp_ocl_compute_ocl_adaptive (unsigned nb_iter)
{
  size_t global[2] = {DIM, kernel_fp[0].h};
  size_t local[2]  = {TILE_W, TILE_H};
  cl_int err;
  uint64_t clock;
  unsigned change = 0;

  for (unsigned iter = 1; iter <= nb_iter; iter++) {
    // gpu
    enqueue_kernel (err, global, local, &clock);
    // cpu
    compute_cpu (&change);
    finish_and_time (clock);
    ocl_swap_tables ();
    if (++true_iter_number % GPU_CPU_SYNC_FREQ == 0 && true_iter_number > 0) {
      ocl_sync_borders (err);
      if (kernel_durations[0]) {
        if (much_greater_than (kernel_durations[0],
                               kernel_durations[1]) &&
            kernel_fp[0].h > TILE_H * 2) { // cpu going faster
          PRINT_DEBUG ('v', "Giving more work to the CPU\n");
          clEnqueueReadBuffer (
              ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0,
              sizeof (cell_t) * DIM * kernel_fp[0].h, _table, 0, NULL,
              NULL); // TODO: make this load only whats required
          kernel_fp[0].h -= TILE_H;
          kernel_fp[1].h += TILE_H;
          kernel_fp[1].y = kernel_fp[0].y + kernel_fp[0].h;
          global[1]      = kernel_fp[0].h;
        } else if (much_greater_than (kernel_durations[1],
                                      kernel_durations[0]) &&
                   kernel_fp[1].h > TILE_H) { // gpu going brrrr fast
          PRINT_DEBUG ('v', "Giving more work to the GPU\n");
          clEnqueueWriteBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE,
                                sizeof (cell_t) * kernel_fp[0].h * DIM,
                                sizeof (cell_t) * DIM * TILE_H,
                                _table + (DIM * kernel_fp[0].h), 0, NULL, NULL);
          kernel_fp[0].h += TILE_H;
          kernel_fp[1].h -= TILE_H;
          kernel_fp[1].y = kernel_fp[0].y + kernel_fp[0].h;
          global[1]      = kernel_fp[0].h;
        }
      }
    }
  }
  return 0;
}

/* === refresh fn === */
void life_omp_ocl_refresh_img_ocl (void)
{
  cl_int err;

  err = clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0,
                             sizeof (cell_t) * DIM *
                                 (kernel_fp[0].h - BORDER_SIZE),
                             _table, 0, NULL, NULL);
  check (err, "Failed to read buffer chunk from GPU");
  life_omp_ocl_refresh_img ();
}
void life_omp_ocl_refresh_img_ocl_adaptive ()
{
  life_omp_ocl_refresh_img_ocl ();
}
void life_omp_ocl_refresh_img_ocl_mt ()
{
  life_omp_ocl_refresh_img_ocl ();
}
#else
/* === First version I try, GPU takes care of the bottom of the table === */
#define TILE_W_OPT 32
#define TILE_H_OPT 8
#define CPU_GPU_SYNC_FREQ 10
#define NB_LINES_FOR_GPU 512
#define BORDER_SIZE CPU_GPU_SYNC_FREQ
static cl_mem gpu_table_ocl = 0, gpu_alternage_table_ocl = 0;
static int nb_iter_true = 0;
// for the first version, we're going to fix the n° of lines computed by the
// CPU vs by the GPU to NB_LINES_FOR_GPU.
// we're going to send a border as well, of size CPU_GPU_SYNC_FREQ-1
// The CPU will have the full table, with part of it out of sync. Every
// CPU_GPU_SYNC_FREQ we're going to sync CPU and GPU.
void life_omp_ocl_init_ocl (void)
{
  life_omp_ocl_init ();
  life_omp_ocl_ft_ocl ();

  const unsigned gpu_size = DIM * NB_LINES_FOR_GPU * sizeof (cell_t);
  gpu_table_ocl =
      clCreateBuffer (context, CL_MEM_READ_WRITE, gpu_size, NULL, NULL);
  if (!gpu_table_ocl)
    exit_with_error ("Failed to allocate gpu_table_ocl buffer");
  gpu_alternage_table_ocl =
      clCreateBuffer (context, CL_MEM_READ_WRITE, gpu_size, NULL, NULL);
  if (!gpu_alternage_table_ocl)
    exit_with_error ("Failed to allocate gpu_alternage_table_ocl buffer");
}

void life_omp_ocl_draw_ocl (char *params)
{
  life_omp_ocl_draw (params);
  const unsigned gpu_size = DIM * NB_LINES_FOR_GPU * sizeof (cell_t);
  cl_int err;
  err = clEnqueueWriteBuffer (ocl_queue (0), gpu_table_ocl, CL_TRUE, 0,
                              gpu_size, _table, 0, NULL, NULL);
  check (err, "Failed to write gpu_table_ocl");
  err = clEnqueueWriteBuffer (ocl_queue (0), gpu_alternage_table_ocl, CL_TRUE,
                              0, gpu_size, _alternate_table, 0, NULL, NULL);
  check (err, "Failed to write gpu_alternage_table_ocl");
}

static inline void compute_gpu (size_t global[2], size_t local[2], cl_int err)
{
  monitoring_start (easypap_gpu_lane (0));
  err = 0;
  err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                         &gpu_table_ocl);
  err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                         &gpu_alternage_table_ocl);
  check (err, "Failed to set kernel computing arguments");
  err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2, NULL,
                                global, local, 0, NULL, NULL);
  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, NB_LINES_FOR_GPU - BORDER_SIZE,
                       easypap_gpu_lane (0));
  check (err, "Failed to execute kernel");
}

static inline void sync_cpu_gpu (cl_int err)
{
  unsigned true_gpu_size =
      sizeof (cell_t) * DIM * (NB_LINES_FOR_GPU - BORDER_SIZE);

  err = clEnqueueReadBuffer (
      ocl_queue (0), gpu_table_ocl, CL_TRUE,
      sizeof (cell_t) * DIM * (NB_LINES_FOR_GPU - BORDER_SIZE * 2),
      sizeof (cell_t) * DIM * BORDER_SIZE,
      _table + DIM * (NB_LINES_FOR_GPU - BORDER_SIZE * 2), 0, NULL, NULL);
  check (err, "Err syncing host to device");

  size_t border_offset_elements = DIM * (NB_LINES_FOR_GPU - BORDER_SIZE);

  err =
      clEnqueueWriteBuffer (ocl_queue (0), gpu_table_ocl, CL_TRUE,
                            true_gpu_size, BORDER_SIZE * DIM * sizeof (cell_t),
                            _table + border_offset_elements, 0, NULL, NULL);
  check (err, "Err syncing device to host");
}

static inline void switch_tables_ocl (void)
{
  cl_mem tmp              = gpu_table_ocl;
  gpu_table_ocl           = gpu_alternage_table_ocl;
  gpu_alternage_table_ocl = tmp;
  cell_t *tmp2            = _table;
  _table                  = _alternate_table;
  _alternate_table        = tmp2;
}

unsigned life_omp_ocl_compute_ocl (unsigned nb_iter)
{

  size_t global[2] = {GPU_SIZE_X, NB_LINES_FOR_GPU};
  size_t local[2]  = {TILE_W_OPT, TILE_H_OPT};
  cl_int err       = 0;

  int change       = 0;
  int border_tiles = (BORDER_SIZE * 2) / TILE_H + 1;
  int cpu_start_y  = NB_LINES_FOR_GPU - (border_tiles * TILE_H);

  for (unsigned it = 1; it <= nb_iter; it++) {
#pragma omp parallel master
    {
#pragma omp task
      {
        compute_gpu (global, local, err);
      }
#pragma omp taskloop collapse(2)
      for (int y = cpu_start_y; y < DIM; y += TILE_H) {
        for (int x = 0; x < DIM; x += TILE_W) {
          change |= do_tile (x, y, TILE_W, TILE_H);
        }
      }
    }

    switch_tables_ocl ();

    if (++nb_iter_true % CPU_GPU_SYNC_FREQ == 0 && nb_iter_true > 0) {
      sync_cpu_gpu (err);
    }
  }
  return 0;
}

void life_omp_ocl_refresh_img_ocl (void)
{
  cl_int err;

  err = clEnqueueReadBuffer (ocl_queue (0), gpu_table_ocl, CL_TRUE, 0,
                             sizeof (cell_t) * DIM *
                                 (NB_LINES_FOR_GPU - BORDER_SIZE),
                             _table, 0, NULL, NULL);
  check (err, "Failed to read buffer chunk from GPU");
  life_omp_ocl_refresh_img ();
}
#endif
#endif

void life_omp_ocl_finalize (void)
{
  unsigned size = DIM * DIM * sizeof (cell_t);
  munmap (_table, size);
  munmap (_alternate_table, size);
}

// This function is called whenever the graphical window needs to be refreshed
void life_omp_ocl_refresh_img (void)
{
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * LIFE_COLOR;
}

static inline void swap_tables (void)
{
  cell_t *tmp = _table;

  _table           = _alternate_table;
  _alternate_table = tmp;
}

///////////////////////////// Default tiling
int life_omp_ocl_do_tile_default (int x, int y, int width, int height)
{
  char change = 0;
  // precomputing start and end indexes of tile's both width and height
  int x_start = (x == 0) ? 1 : x;
  int x_end   = (x + width >= DIM) ? DIM - 1 : x + width;
  int y_start = (y == 0) ? 1 : y;
  int y_end   = (y + height >= DIM) ? DIM - 1 : y + height;

  for (int i = y_start; i < y_end; i++) {
    for (int j = x_start; j < x_end; j++) {
      const char me = cur_table (i, j);
      // we unrolled the loop and check it in lines
      const char n = cur_table (i - 1, j - 1) + cur_table (i - 1, j) +
                     cur_table (i - 1, j + 1) + cur_table (i, j - 1) +
                     cur_table (i, j + 1) + cur_table (i + 1, j - 1) +
                     cur_table (i + 1, j) + cur_table (i + 1, j + 1);
      // while we are at it, we apply some simple branchless programming logic
      const char new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
      change |= (me ^ new_me);
      next_table (i, j) = new_me;
    }
  }
  return change;
}

///////////////////////////// Sequential version (seq)
unsigned life_omp_ocl_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    int change = do_tile (0, 0, DIM, DIM);

    if (!change)
      return it;

    swap_tables ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
//
unsigned life_omp_ocl_compute_tiled (unsigned nb_iter)
{
  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = do_tile (0, 0, DIM, DIM);

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |= do_tile (x, y, TILE_W, TILE_H);

    if (!change)
      return it;

    swap_tables ();
  }

  return res;
}
///////////////////////////// ompfor version

unsigned life_omp_ocl_compute_ompfor (unsigned nb_iter)
{
  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W) {
        change |= do_tile (x, y, TILE_W, TILE_H);
      }

    swap_tables ();

    if (!change) { // we stop if all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

///////////////////////////// First touch allocations
void life_omp_ocl_ft (void)
{
#pragma omp parallel for schedule(runtime) collapse(2)
  for (int y = 0; y < DIM; y += TILE_H)
    for (int x = 0; x < DIM; x += TILE_W) {
      next_table (y, x) = cur_table (y, x) = 0;
    }
}
void life_omp_ocl_ft_ocl (void)
{
  life_omp_ocl_ft ();
}
///////////////////////////// Initial configs

void life_omp_ocl_draw_guns (void);

static inline void set_cell (int y, int x)
{
  cur_table (y, x) = 1;
  if (gpu_used)
    *((char *)image + y * DIM + x) = 1;
}

static inline int get_cell (int y, int x)
{
  return cur_table (y, x);
}

static void inline life_omp_ocl_rle_parse (char *filename, int x, int y,
                                           int orientation)
{
  rle_lexer_parse (filename, x, y, set_cell, orientation);
}

static void inline life_omp_ocl_rle_generate (char *filename, int x, int y,
                                              int width, int height)
{
  rle_generate (x, y, width, height, get_cell, filename);
}

void life_omp_ocl_draw (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_omp_ocl_rle_parse (param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper (param, life_omp_ocl_draw_guns);
}

static void otca_autoswitch (char *name, int x, int y)
{
  life_omp_ocl_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_omp_ocl_rle_parse ("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                          RLE_ORIENTATION_NORMAL);
}

static void otca_life_omp_ocl (char *name, int x, int y)
{
  life_omp_ocl_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_omp_ocl_rle_parse ("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                          RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners (char *filename, int distance)
{
  life_omp_ocl_rle_parse (filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_omp_ocl_rle_parse (filename, distance, distance,
                          RLE_ORIENTATION_HINVERT);
  life_omp_ocl_rle_parse (filename, distance, distance,
                          RLE_ORIENTATION_VINVERT);
  life_omp_ocl_rle_parse (filename, distance, distance,
                          RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life_omp_ocl -s 2176 -a otca_off -ts 64 -r 10 -si
void life_omp_ocl_draw_otca_off (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_omp_ocl -s 2176 -a otca_on -ts 64 -r 10 -si
void life_omp_ocl_draw_otca_on (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_omp_ocl -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_omp_ocl_draw_meta3x3 (void)
{
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life_omp_ocl (j == 1 ? "data/rle/otca-on.rle"
                                : "data/rle/otca-off.rle",
                         1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life_omp_ocl -a bugs -ts 64
void life_omp_ocl_draw_bugs (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_omp_ocl_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                            RLE_ORIENTATION_NORMAL);
    life_omp_ocl_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                            RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life_omp_ocl -v omp -a ship -s 512 -m -ts 16
void life_omp_ocl_draw_ship (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_omp_ocl_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                            RLE_ORIENTATION_NORMAL);
    life_omp_ocl_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                            RLE_ORIENTATION_NORMAL);
  }

  for (int y = 43; y < DIM - 134; y += 148) {
    life_omp_ocl_rle_parse ("data/rle/greyship.rle", DIM - 100, y,
                            RLE_ORIENTATION_NORMAL);
  }
}

void life_omp_ocl_draw_stable (void)
{
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell (i, j);
      set_cell (i, j + 1);
      set_cell (i + 1, j);
      set_cell (i + 1, j + 1);
    }
}

void life_omp_ocl_draw_oscil (void)
{
  for (int i = 2; i < DIM - 4; i += 4)
    for (int j = 2; j < DIM - 4; j += 4) {
      if ((j - 2) % 8) {
        set_cell (i + 1, j);
        set_cell (i + 1, j + 1);
        set_cell (i + 1, j + 2);
      } else {
        set_cell (i, j + 1);
        set_cell (i + 1, j + 1);
        set_cell (i + 2, j + 1);
      }
    }
}

void life_omp_ocl_draw_guns (void)
{
  at_the_four_corners ("data/rle/gun.rle", 1);
}

static unsigned long seed = 123456789;

// Deterministic function to generate pseudo-random configurations
// independently of the call context
static unsigned long pseudo_random ()
{
  unsigned long a = 1664525;
  unsigned long c = 1013904223;
  unsigned long m = 4294967296;

  seed = (a * seed + c) % m;
  seed ^= (seed >> 21);
  seed ^= (seed << 35);
  seed ^= (seed >> 4);
  seed *= 2685821657736338717ULL;
  return seed;
}

void life_omp_ocl_draw_random (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (pseudo_random () & 1)
        set_cell (i, j);
}

// Suggested cmdline: ./run -k life_omp_ocl -a clown -s 256 -i 110
void life_omp_ocl_draw_clown (void)
{
  life_omp_ocl_rle_parse ("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                          RLE_ORIENTATION_NORMAL);
}

void life_omp_ocl_draw_diehard (void)
{
  life_omp_ocl_rle_parse ("data/rle/diehard.rle", DIM / 2, DIM / 2,
                          RLE_ORIENTATION_NORMAL);
}

static void dump (int size, int x, int y)
{
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      if (get_cell (i, j))
        set_cell (i + x, j + y);
}

static void moult_rle (int size, int p, char *filepath)
{

  int positions = (DIM) / (size + 1);

  life_omp_ocl_rle_parse (filepath, size / 2, size / 2, RLE_ORIENTATION_NORMAL);
  for (int k = 0; k < p; k++) {
    int px = pseudo_random () % positions;
    int py = pseudo_random () % positions;
    dump (size, px * size, py * size);
  }
}

// ./run  -k life_omp_ocl -a moultdiehard130  -v omp -ts 32 -m -s 512
void life_omp_ocl_draw_moultdiehard130 (void)
{
  moult_rle (16, 128, "data/rle/diehard.rle");
}

// ./run  -k life_omp_ocl -a moultdiehard2474  -v omp -ts 32 -m -s 1024
void life_omp_ocl_draw_moultdiehard1398 (void)
{
  moult_rle (52, 96, "data/rle/diehard1398.rle");
}

// ./run  -k life_omp_ocl -a moultdiehard2474  -v omp -ts 32 -m -s 2048
void life_omp_ocl_draw_moultdiehard2474 (void)
{
  moult_rle (104, 32, "data/rle/diehard2474.rle");
}

// Just in case we want to draw an initial configuration and dump it to file,
// with no iteration at all
unsigned life_omp_ocl_compute_none (unsigned nb_iter)
{
  return 1;
}

//////////// debug ////////////
static int debug_hud = -1;

void life_omp_ocl_config (char *param)
{
  seed += param ? atoi (param) : 0;
  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], debug_hud);
  }
}

void life_omp_ocl_debug (int x, int y)
{
  if (x == -1 || y == -1)
    ezv_hud_set (ctx[0], debug_hud, NULL);
  else {
    ezv_hud_set (ctx[0], debug_hud, cur_table (y, x) ? "Alive" : "Dead");
  }
}
