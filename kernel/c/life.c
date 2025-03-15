
#include "easypap.h"
#include "rle_lexer.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define LIFE_COLOR (ezv_rgb (255, 255, 0))

typedef bool cell_t;

static cell_t *restrict _table           = NULL;
static cell_t *restrict _alternate_table = NULL;

static cell_t *restrict _dirty_tiles     = NULL;
static cell_t *restrict _dirty_tiles_alt = NULL;

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  return i + y * DIM + x;
}

static inline cell_t *dirty_cell (cell_t *restrict i, int y, int x)
{
  return i + (y+1) * (DIM/TILE_W) + (x+1);
}


// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

// using a bordered array in order to be able to do out of bound writes seemlessly.
// must be faster than doing boundary checks
#define cur_dirty(y, x) (*dirty_cell (_dirty_tiles, (y), (x)))
#define next_dirty(y, x) (*dirty_cell (_dirty_tiles_alt, (y), (x)))

void life_init (void)
{
  // life_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    unsigned size = DIM * DIM * sizeof (cell_t);

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d ", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    size = (DIM/TILE_W + 2) * (DIM/TILE_H + 2) * sizeof(cell_t);

    PRINT_DEBUG('u', " + 2x %d bytes\n", size);

    _dirty_tiles = mmap (NULL, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    _dirty_tiles_alt = mmap (NULL, size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    memset(_dirty_tiles, 1, size);
    memset(_dirty_tiles_alt, 0, size);
  }
}

void life_finalize (void)
{
  unsigned size = DIM * DIM * sizeof (cell_t);
  munmap (_table, size);
  munmap (_alternate_table, size);

  size = (DIM/TILE_W + 2) * (DIM/TILE_H + 2) * sizeof(cell_t);

  munmap(_dirty_tiles, size);
  munmap(_dirty_tiles_alt, size);
}

// This function is called whenever the graphical window needs to be refreshed
void life_refresh_img (void)
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

static inline void swap_tables_w_dirty (void)
{
  cell_t *tmp = _table;
  cell_t *tmp2 = _dirty_tiles;

  _table           = _alternate_table;
  _alternate_table = tmp;

  _dirty_tiles = _dirty_tiles_alt;
  _dirty_tiles_alt = tmp2;
}

///////////////////////////// Default tiling
int life_do_tile_default (int x, int y, int width, int height)
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

///////////////////////////// Do tile optimized
int life_do_tile_opt (const int x, const int y, const int width, const int height)
{
  register char change  = 0;
  // precomputing start and end indexes of tile's both width and height
  int x_start = (x == 0) ? 1 : x;
  int x_end   = (x + width >= DIM) ? DIM - 1 : x + width;
  int y_start = (y == 0) ? 1 : y;
  int y_end   = (y + height >= DIM) ? DIM - 1 : y + height;

  for (int i = y_start; i < y_end; i++) {
    for (int j = x_start; j < x_end; j++) {
      const char me = cur_table (i, j);

      // uint32_t top = *(uint32_t*)(cur_table(i-1, j-1)) & 0x00FFFFFF;
      // uint32_t mid = *(uint32_t*)(cur_table(i,   j-1)) & 0x00FFFFFF;
      // uint32_t bot = *(uint32_t*)(cur_table(i+1, j-1)) & 0x00FFFFFF;

      // uint32_t neighborhood = (top << 16) | (mid << 8) | bot;

      // //neighborhood &= ~(1 << 8);

      // int n = __builtin_popcount(neighborhood); // yet for now its way slower

       const char n = cur_table(i-1, j-1) + cur_table(i-1, j) + cur_table(i-1, j+1)
               + cur_table(i, j-1) + cur_table(i, j+1) + cur_table(i+1, j-1)
         + cur_table(i+1, j) + cur_table(i+1, j+1);
      // while we are at it, let's apply some simple branchless programming logic
      const char new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
      change |= (me ^ new_me);
      next_table (i, j) = new_me;
    }
  }

  return change;
}

///////////////////////////// Sequential version (seq)
//
unsigned life_compute_seq (unsigned nb_iter)
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
unsigned life_compute_tiled (unsigned nb_iter)
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

///////////////////////////// ompfor  version
//./run -k life -v ompfor -ts 64 -a moultdiehard130 -m
unsigned life_compute_omp_tiled (unsigned nb_iter)
{
  unsigned res = 0;

#pragma omp parallel
  {
    unsigned local_change = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
      local_change = do_tile (0, 0, DIM, DIM);

#pragma omp for collapse(2) schedule(runtime) nowait
      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          local_change |=
              do_tile (x, y, TILE_W, TILE_H); // Combine changes from all tiles

#pragma omp single
      {
        if (!local_change) { // If no changes, stop early
          res = it;
          it  = nb_iter + 1; // Ensure all threads exit loop
        }
        swap_tables ();
      }
    }
  }

  return res;
}

unsigned life_compute_lazy_ompfor(unsigned nb_iter) {
  unsigned res = 0;
  unsigned tile_x, tile_y;
  for (int it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
#pragma omp parallel for schedule(runtime) collapse(2) reduction(|: change) private(tile_x, tile_y)
    for (int y = 0; y < DIM; y += TILE_H) {
      for (int x = 0; x < DIM; x += TILE_W) {
        tile_y = y / TILE_H;
        unsigned local_change = 0;
        tile_x = x / TILE_W;
        if (cur_dirty(tile_y, tile_x)) {
          local_change = do_tile(x, y, TILE_W, TILE_H);
          change |= local_change;

          if (local_change) {
            next_dirty(tile_y-1, tile_x-1) = 1;
            next_dirty(tile_y-1, tile_x)   = 1;
            next_dirty(tile_y-1, tile_x+1) = 1;
            next_dirty(tile_y, tile_x-1)   = 1;
            next_dirty(tile_y, tile_x)     = 1; 
            next_dirty(tile_y, tile_x+1)   = 1;
            next_dirty(tile_y+1, tile_x-1) = 1;
            next_dirty(tile_y+1, tile_x)   = 1;
            next_dirty(tile_y+1, tile_x+1) = 1;
          } else {
            cur_dirty(tile_y, tile_x) = 0;
          }
        }
      }
    }
    if(!change) return it;
    swap_tables_w_dirty();
  }

  return res;
}

unsigned life_compute_lazy(unsigned nb_iter) {
  unsigned res = 0;

  for (int it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    for (int y = 0; y < DIM; y += TILE_H) {
      unsigned tile_y = y / TILE_H;
      for (int x = 0; x < DIM; x += TILE_W) {
        unsigned local_change = 0;
        unsigned tile_x = x / TILE_W;
        if (cur_dirty(tile_y, tile_x)) {
          local_change = do_tile(x, y, TILE_W, TILE_H);
          change |= local_change;

          if (local_change) {
            next_dirty(tile_y-1, tile_x-1) = 1;
            next_dirty(tile_y-1, tile_x)   = 1;
            next_dirty(tile_y-1, tile_x+1) = 1;
            next_dirty(tile_y, tile_x-1)   = 1;
            next_dirty(tile_y, tile_x)     = 1; 
            next_dirty(tile_y, tile_x+1)   = 1;
            next_dirty(tile_y+1, tile_x-1) = 1;
            next_dirty(tile_y+1, tile_x)   = 1;
            next_dirty(tile_y+1, tile_x+1) = 1;
          } else {
            cur_dirty(tile_y, tile_x) = 0;
          }
        }
      }
    }
    if(!change) return it;
    swap_tables_w_dirty();
  }

  return res;
}

///////////////////////////// Tiled ompfor version
//
unsigned life_compute_ompfor (unsigned nb_iter)
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

///////////////////////////// Tiled taskloop version
//

unsigned life_compute_omptaskloop (unsigned nb_iter)
{
  unsigned res = 0;

#pragma omp parallel
#pragma omp single
  {

    for (unsigned it = 1; it <= nb_iter; it++) {
      unsigned change = 0;

#pragma omp taskgroup
      {
#pragma omp taskloop collapse(2) reduction(| : change) grainsize(4)
        for (int y = 0; y < DIM; y += TILE_H)
          for (int x = 0; x < DIM; x += TILE_W) {
            change |= do_tile (x, y, TILE_W, TILE_H);
          }
      }
      swap_tables ();

      if (!change) { // we stop if all cells are stable
        res = it;
        break;
      }
    }
  }

  return res;
}


///////////////////////////// Initial configs

void life_draw_guns (void);

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

static void inline life_rle_parse (char *filename, int x, int y,
                                   int orientation)
{
  rle_lexer_parse (filename, x, y, set_cell, orientation);
}

static void inline life_rle_generate (char *filename, int x, int y, int width,
                                      int height)
{
  rle_generate (x, y, width, height, get_cell, filename);
}

void life_draw (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_rle_parse (param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper (param, life_draw_guns);
}

static void otca_autoswitch (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void otca_life (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners (char *filename, int distance)
{
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_HINVERT);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_VINVERT);
  life_rle_parse (filename, distance, distance,
                  RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_off -ts 64 -r 10 -si
void life_draw_otca_off (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_on -ts 64 -r 10 -si
void life_draw_otca_on (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_draw_meta3x3 (void)
{
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life (j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                 1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life -a bugs -ts 64
void life_draw_bugs (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life -v omp -a ship -s 512 -m -ts 16
void life_draw_ship (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }

  for (int y = 43; y < DIM - 134; y += 148) {
    life_rle_parse ("data/rle/greyship.rle", DIM - 100, y,
                    RLE_ORIENTATION_NORMAL);
  }
}

void life_draw_stable (void)
{
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell (i, j);
      set_cell (i, j + 1);
      set_cell (i + 1, j);
      set_cell (i + 1, j + 1);
    }
}

void life_draw_oscil (void)
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

void life_draw_guns (void)
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

void life_draw_random (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (pseudo_random () & 1)
        set_cell (i, j);
}

// Suggested cmdline: ./run -k life -a clown -s 256 -i 110
void life_draw_clown (void)
{
  life_rle_parse ("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}

void life_draw_diehard (void)
{
  life_rle_parse ("data/rle/diehard.rle", DIM / 2, DIM / 2,
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

  life_rle_parse (filepath, size / 2, size / 2, RLE_ORIENTATION_NORMAL);
  for (int k = 0; k < p; k++) {
    int px = pseudo_random () % positions;
    int py = pseudo_random () % positions;
    dump (size, px * size, py * size);
  }
}

// ./run  -k life -a moultdiehard130  -v omp -ts 32 -m -s 512
void life_draw_moultdiehard130 (void)
{
  moult_rle (16, 128, "data/rle/diehard.rle");
}

// ./run  -k life -a moultdiehard2474  -v omp -ts 32 -m -s 1024
void life_draw_moultdiehard1398 (void)
{
  moult_rle (52, 96, "data/rle/diehard1398.rle");
}

// ./run  -k life -a moultdiehard2474  -v omp -ts 32 -m -s 2048
void life_draw_moultdiehard2474 (void)
{
  moult_rle (104, 32, "data/rle/diehard2474.rle");
}

// Just in case we want to draw an initial configuration and dump it to file,
// with no iteration at all
unsigned life_compute_none (unsigned nb_iter)
{
  return 1;
}

//////////// debug ////////////
static int debug_hud = -1;

void life_config (char *param)
{
  seed += param ? atoi (param) : 0;
  if (picking_enabled) {
    debug_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], debug_hud);
  }
}

void life_debug (int x, int y)
{
  if (x == -1 || y == -1)
    ezv_hud_set (ctx[0], debug_hud, NULL);
  else {
    ezv_hud_set (ctx[0], debug_hud, cur_table (y, x) ? "Alive" : "Dead");
  }
}
