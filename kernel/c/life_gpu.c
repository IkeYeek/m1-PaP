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

void life_gpu_init (void)
{
  // life_gpu_init may be (indirectly) called several times so we check if data
  // were already allocated
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
static cl_mem tile_in = 0, tile_out = 0;
void life_gpu_init_ocl_lazy (void)
{
  life_gpu_init ();
  const unsigned size =
      ((DIM / TILE_W) + 2) * ((DIM / TILE_H) + 2) * sizeof (cell_t);
  tile_in = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!tile_in)
    exit_with_error ("Failed to allocate tile_in buffer");
  tile_out = clCreateBuffer (context, CL_MEM_READ_WRITE, size, NULL, NULL);
  if (!tile_out)
    exit_with_error ("Failed to allocate tile_out buffer");
}
void life_gpu_alloc_buffers_ocl (void)
{
  printf ("lol\n");
}
void life_gpu_draw_ocl_lazy (char *params)
{
  life_gpu_draw (params);
  const unsigned size =
      ((DIM / TILE_W) + 2) * ((DIM / TILE_H) + 2) * sizeof (cell_t);
  cl_int err;
  cell_t *all_1 = malloc (size);
  cell_t *all_0 = malloc (size);
  for (int y = 0; y < (DIM / TILE_H) + 2; y++) {
    for (int x = 0; x < (DIM / TILE_W) + 2; x++) {
      all_1[y * (DIM / TILE_W) + x] = 1;
      all_0[y * (DIM / TILE_W) + x] = 0;
    }
  }

  err = clEnqueueWriteBuffer (ocl_queue (0), tile_in, CL_TRUE, 0, size, all_1,
                              0, NULL, NULL);
  check (err, "Failed to write to tile_in");
  err = clEnqueueWriteBuffer (ocl_queue (0), tile_out, CL_TRUE, 0, size, all_0,
                              0, NULL, NULL);
  check (err, "Failed to write to tile_out");

  free (all_0);
  free (all_1);
}

unsigned life_gpu_compute_ocl_lazy (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local[2]  = {TILE_W, TILE_H};
  cl_int err;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it++) {
    err = 0;
    // cl_kernel reset_tile_out = clCreateKernel (program, "reset_tile_out",
    // &err); check (err, "Failed to load reset kernel arguments"); err |=
    // clSetKernelArg (reset_tile_out, 0, sizeof (cl_mem), &tile_out);
    //
    // check (err, "Failed to set tile_out reset kernel arguments");
    //
    // err = clEnqueueNDRangeKernel (ocl_queue (0), reset_tile_out, 2, NULL,
    //                               global, local, 0, NULL, NULL);
    // clFinish (ocl_queue (0));

    err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                           &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    err |=
        clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (cl_mem), &tile_in);
    err |=
        clSetKernelArg (ocl_compute_kernel (0), 3, sizeof (cl_mem), &tile_out);
    check (err, "Failed to set kernel computing arguments");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");
    clFinish (ocl_queue (0));
    {
      cl_mem tmp          = ocl_next_buffer (0);
      ocl_next_buffer (0) = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = tmp;
      tmp                 = tile_in;
      tile_in             = tile_out;
      tile_out            = tmp;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}
unsigned life_gpu_compute_ocl_2x (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X / 2, GPU_SIZE_Y};
  size_t local[2]  = {TILE_W, TILE_H};
  cl_int err;
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it++) {
    err = 0;
    err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                           &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");
    {
      cl_mem tmp          = ocl_next_buffer (0);
      ocl_next_buffer (0) = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = tmp;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}

unsigned ilog2 (unsigned int x)
{
  unsigned log = 0;
  while (x >>= 1)
    ++log;
  return log;
}

unsigned life_gpu_compute_ocl_binmul (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local[2]  = {TILE_W, TILE_H};
  cl_int err;
  unsigned ilog = ilog2 (DIM);
  monitoring_start (easypap_gpu_lane (0));

  for (unsigned it = 1; it <= nb_iter; it++) {
    err = 0;
    err |= clSetKernelArg (ocl_compute_kernel (0), 0, sizeof (cl_mem),
                           &ocl_cur_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 1, sizeof (cl_mem),
                           &ocl_next_buffer (0));
    err |= clSetKernelArg (ocl_compute_kernel (0), 2, sizeof (unsigned), &ilog);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (ocl_queue (0), ocl_compute_kernel (0), 2,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");
    {
      cl_mem tmp          = ocl_next_buffer (0);
      ocl_next_buffer (0) = ocl_cur_buffer (0);
      ocl_cur_buffer (0)  = tmp;
    }
  }

  clFinish (ocl_queue (0));
  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (0));
  return 0;
}
void life_gpu_refresh_img_ocl (void)
{
  // TODO: adapt this when i will have some graphical display
  // got some serious helps from claude with this one because i had no clue what
  // was happening
  cl_int err;

  err =
      clEnqueueReadBuffer (ocl_queue (0), ocl_cur_buffer (0), CL_TRUE, 0,
                           sizeof (cell_t) * DIM * DIM, _table, 0, NULL, NULL);
  check (err, "Failed to read buffer chunk from GPU");

  life_gpu_refresh_img ();
}

void life_gpu_refresh_img_ocl_localmem (void)
{
  life_gpu_refresh_img_ocl ();
}
void life_gpu_refresh_img_ocl_2x (void)
{
  life_gpu_refresh_img_ocl ();
}
void life_gpu_refresh_img_ocl_binlum (void)
{
  life_gpu_refresh_img_ocl ();
}
#endif

void life_gpu_finalize (void)
{
  unsigned size = (DIM + 1) * (DIM + 1) * sizeof (cell_t);
  munmap (_table, size);
  munmap (_alternate_table, size);
}

// This function is called whenever the graphical window needs to be refreshed
void life_gpu_refresh_img (void)
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
int life_gpu_do_tile_default (int x, int y, int width, int height)
{
  int change = 0;
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      if (j > 0 && j < DIM - 1 && i > 0 && i < DIM - 1) {

        unsigned n  = 0;
        unsigned me = cur_table (i, j);

        for (int yloc = i - 1; yloc < i + 2; yloc++)
          for (int xloc = j - 1; xloc < j + 2; xloc++)
            if (xloc != j || yloc != i)
              n += cur_table (yloc, xloc);

        if (me == 1 && n != 2 && n != 3) {
          me     = 0;
          change = 1;
        } else if (me == 0 && n == 3) {
          me     = 1;
          change = 1;
        }

        next_table (i, j) = me;
      }
  return change;
}

///////////////////////////// Sequential version (seq)
unsigned life_gpu_compute_seq (unsigned nb_iter)
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
unsigned life_gpu_compute_tiled (unsigned nb_iter)
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

///////////////////////////// First touch allocations
void life_gpu_ft (void)
{
#pragma omp parallel for schedule(runtime) collapse(2)
  for (int y = 0; y < DIM; y += TILE_H)
    for (int x = 0; x < DIM; x += TILE_W) {
      next_table (y, x) = cur_table (y, x) = 0;
    }
}

///////////////////////////// Initial configs

void life_gpu_draw_guns (void);

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

static void inline life_gpu_rle_parse (char *filename, int x, int y,
                                       int orientation)
{
  rle_lexer_parse (filename, x, y, set_cell, orientation);
}

static void inline life_gpu_rle_generate (char *filename, int x, int y,
                                          int width, int height)
{
  rle_generate (x, y, width, height, get_cell, filename);
}

void life_gpu_draw (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_gpu_rle_parse (param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper (param, life_gpu_draw_guns);
}

static void otca_autoswitch (char *name, int x, int y)
{
  life_gpu_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_gpu_rle_parse ("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                      RLE_ORIENTATION_NORMAL);
}

static void otca_life_gpu (char *name, int x, int y)
{
  life_gpu_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_gpu_rle_parse ("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                      RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners (char *filename, int distance)
{
  life_gpu_rle_parse (filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_gpu_rle_parse (filename, distance, distance, RLE_ORIENTATION_HINVERT);
  life_gpu_rle_parse (filename, distance, distance, RLE_ORIENTATION_VINVERT);
  life_gpu_rle_parse (filename, distance, distance,
                      RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life_gpu -s 2176 -a otca_off -ts 64 -r 10 -si
void life_gpu_draw_otca_off (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_gpu -s 2176 -a otca_on -ts 64 -r 10 -si
void life_gpu_draw_otca_on (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life_gpu -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_gpu_draw_meta3x3 (void)
{
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life_gpu (j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                     1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life_gpu -a bugs -ts 64
void life_gpu_draw_bugs (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_gpu_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                        RLE_ORIENTATION_NORMAL);
    life_gpu_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                        RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life_gpu -v omp -a ship -s 512 -m -ts 16
void life_gpu_draw_ship (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_gpu_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                        RLE_ORIENTATION_NORMAL);
    life_gpu_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                        RLE_ORIENTATION_NORMAL);
  }

  for (int y = 43; y < DIM - 134; y += 148) {
    life_gpu_rle_parse ("data/rle/greyship.rle", DIM - 100, y,
                        RLE_ORIENTATION_NORMAL);
  }
}

void life_gpu_draw_stable (void)
{
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell (i, j);
      set_cell (i, j + 1);
      set_cell (i + 1, j);
      set_cell (i + 1, j + 1);
    }
}

void life_gpu_draw_oscil (void)
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

void life_gpu_draw_guns (void)
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

void life_gpu_draw_random (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (pseudo_random () & 1)
        set_cell (i, j);
}

// Suggested cmdline: ./run -k life_gpu -a clown -s 256 -i 110
void life_gpu_draw_clown (void)
{
  life_gpu_rle_parse ("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                      RLE_ORIENTATION_NORMAL);
}

void life_gpu_draw_diehard (void)
{
  life_gpu_rle_parse ("data/rle/diehard.rle", DIM / 2, DIM / 2,
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

  life_gpu_rle_parse (filepath, size / 2, size / 2, RLE_ORIENTATION_NORMAL);
  for (int k = 0; k < p; k++) {
    int px = pseudo_random () % positions;
    int py = pseudo_random () % positions;
    dump (size, px * size, py * size);
  }
}

// ./run  -k life_gpu -a moultdiehard130  -v omp -ts 32 -m -s 512
void life_gpu_draw_moultdiehard130 (void)
{
  moult_rle (16, 128, "data/rle/diehard.rle");
}

// ./run  -k life_gpu -a moultdiehard2474  -v omp -ts 32 -m -s 1024
void life_gpu_draw_moultdiehard1398 (void)
{
  moult_rle (52, 96, "data/rle/diehard1398.rle");
}

// ./run  -k life_gpu -a moultdiehard2474  -v omp -ts 32 -m -s 2048
void life_gpu_draw_moultdiehard2474 (void)
{
  moult_rle (104, 32, "data/rle/diehard2474.rle");
}

// Just in case we want to draw an initial configuration and dump it to file,
// with no iteration at all
unsigned life_gpu_compute_none (unsigned nb_iter)
{
  return 1;
}

//////////// debug ////////////
static int debug_hud = -1;

void life_gpu_config (char *param)
{
  seed += param ? atoi (param) : 0;
  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], debug_hud);
  }
}

void life_gpu_debug (int x, int y)
{
  if (x == -1 || y == -1)
    ezv_hud_set (ctx[0], debug_hud, NULL);
  else {
    ezv_hud_set (ctx[0], debug_hud, cur_table (y, x) ? "Alive" : "Dead");
  }
}
