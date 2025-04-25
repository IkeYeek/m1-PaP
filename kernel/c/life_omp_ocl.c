#include "easypap.h"
#include "rle_lexer.h"

#include <CL/cl.h>
#include <numa.h>
#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define LIFE_COLOR (ezv_rgb (255, 255, 0))

typedef unsigned cell_t;

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
/* === First version I try, GPU takes care of the bottom of the table === */
#define CPU_GPU_SYNC_FREQ 10
#define NB_LINES_FOR_GPU 1024
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
unsigned life_omp_ocl_compute_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, NB_LINES_FOR_GPU};
  size_t local[2]  = {TILE_W, TILE_H};
  cl_int err;
  int change = 0;

  omp_set_max_active_levels (2);

  for (unsigned it = 1; it <= nb_iter; it++) {
#pragma omp parallel num_threads(2)
    {
      int thread_id = omp_get_thread_num ();
      if (thread_id == 0) {
        monitoring_start (easypap_gpu_lane (0));
        err = 0;
        err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                               &gpu_table_ocl);
        err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                               &gpu_alternage_table_ocl);
        check (err, "Failed to set kernel computing arguments");
        err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                      NULL, global, local, 0, NULL, NULL);
        clFinish (ocl_queue (0));
        monitoring_end_tile (0, 0, DIM, NB_LINES_FOR_GPU - BORDER_SIZE,
                             easypap_gpu_lane (0));
        check (err, "Failed to execute kernel");
      } else {
        {
          int border_tiles = (BORDER_SIZE * 2) / TILE_H + 1;
          int cpu_start_y  = NB_LINES_FOR_GPU - (border_tiles * TILE_H);

#pragma omp parallel for schedule(runtime) collapse(2)
          for (int y = cpu_start_y; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
              change |= do_tile (x, y, TILE_W, TILE_H);
            }
          }
        }
      }
    }

    cl_mem tmp              = gpu_table_ocl;
    gpu_table_ocl           = gpu_alternage_table_ocl;
    gpu_alternage_table_ocl = tmp;
    cell_t *tmp2            = _table;
    _table                  = _alternate_table;
    _alternate_table        = tmp2;

    if (++nb_iter_true % CPU_GPU_SYNC_FREQ == 0) {
      unsigned true_gpu_size =
          sizeof (cell_t) * DIM * (NB_LINES_FOR_GPU - BORDER_SIZE);
      cl_int err;

      printf ("%d\n", nb_iter_true);
      err = clEnqueueReadBuffer (ocl_queue (0), gpu_table_ocl, CL_TRUE, 0,
                                 true_gpu_size, _table, 0, NULL, NULL);
      check (err, "Err syncing host to device");

      size_t border_offset_elements = DIM * (NB_LINES_FOR_GPU - BORDER_SIZE);

      err = clEnqueueWriteBuffer (
          ocl_queue (0), gpu_table_ocl, CL_TRUE, true_gpu_size,
          BORDER_SIZE * DIM * sizeof (cell_t), _table + border_offset_elements,
          0, NULL, NULL);
      check (err, "Err syncing device to host");
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

///////////////////////////// Initial configs

void life_omp_ocl_draw_guns (void);

static inline void set_cell (int y, int x)
{
  cur_table (y, x) = 1;
  if (gpu_used)
    cur_img (y, x) = 1;
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
