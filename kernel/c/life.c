#include "easypap.h"
#include "rle_lexer.h"

#include <numa.h>
#include <omp.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define LIFE_COLOR (ezv_rgb (255, 255, 0))

typedef char cell_t;

static cell_t *restrict __attribute__ ((aligned (32))) _table           = NULL;
static cell_t *restrict __attribute__ ((aligned (32))) _alternate_table = NULL;
static cell_t *restrict __attribute__ ((aligned (32))) _dirty_tiles     = NULL;
static cell_t *restrict __attribute__ ((aligned (32))) _dirty_tiles_alt = NULL;

static unsigned __attribute__ ((aligned (64))) DIM_PER_TILE_W;

static int rank, size;

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  return i + (y + 1) * DIM + (x + 1);
}

static inline cell_t *dirty_cell (cell_t *restrict i, int y, int x)
{
  return i + (y + 1) * DIM_PER_TILE_W + (x + 1);
}

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

// using a bordered array in order to be able to do out of bound writes
// seemlessly. must be faster than doing boundary checks
#define cur_dirty(y, x) (*dirty_cell (_dirty_tiles, (y), (x)))
#define next_dirty(y, x) (*dirty_cell (_dirty_tiles_alt, (y), (x)))

void life_init (void)
{
  // life_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    unsigned size  = (DIM + 2) * (DIM + 2) * sizeof (cell_t);
    DIM_PER_TILE_W = (DIM / TILE_W);

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d ", size);

    // _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
    //                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    // _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
    //                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    _alternate_table = aligned_alloc (32, size);
    // adding 1 ghost cell on each side in order to allow oob writes by 1. those
    // well never be read anyway
    size = (DIM / TILE_W + 2) * (DIM / TILE_H + 2) * sizeof (cell_t);

    PRINT_DEBUG ('u', " + 2x %d bytes\n", size);

    _dirty_tiles     = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    _dirty_tiles_alt = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    // setting both arrays to 1 so we force at least 1 full board evaluation
    memset (_dirty_tiles, 1, size);
    memset (_dirty_tiles_alt, 1, size);
  }
}

void life_finalize (void)
{
  unsigned size = (DIM + 2) * (DIM + 2) * sizeof (cell_t);
  munmap (_table, size);
  munmap (_alternate_table, size);

  size = (DIM / TILE_W + 2) * (DIM / TILE_H + 2) * sizeof (cell_t);

  munmap (_dirty_tiles, size);
  munmap (_dirty_tiles_alt, size);
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
  cell_t *tmp  = _table;
  cell_t *tmp2 = _dirty_tiles;

  _table           = _alternate_table;
  _alternate_table = tmp;

  unsigned size    = (DIM / TILE_W + 2) * (DIM / TILE_H + 2) * sizeof (cell_t);
  _dirty_tiles     = _dirty_tiles_alt;
  _dirty_tiles_alt = tmp2;
  memset (_dirty_tiles_alt, 0, size);
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
/*
#define ENABLE_VECTO
#define __AVX2__ 1
#define __AVX512__ 1
#ifdef ENABLE_VECTO
#include <immintrin.h>
#if __AVX2__ == 1
// define a macro to factorize vector loading operations
#define M256I_LOADU(y, x)                                                      \
  _mm256_loadu_si256 ((const __m256i *)table_cell (_table, y, x))

__m256i shift_bytes_left (__m256i a)
{
  __m256i zero = _mm256_setzero_si256 ();
  return _mm256_alignr_epi8 (a, zero, 15); // 16-1=15
}

__m256i shift_bytes_right (__m256i a)
{
  __m256i zero = _mm256_setzero_si256 ();
  return _mm256_alignr_epi8 (zero, a, 1);
}

int life_do_tile_avx2_firstidea (const int x, const int y, const int width,
                                 const int height)
{
  char change = 0;
  // precomputing start and end indexes of tile's both width and height
  int x_start = (x == 0) ? 1 : x;
  int x_end   = (x + width >= DIM) ? DIM - 1 : x + width;
  int y_start = (y == 0) ? 1 : y;
  int y_end   = (y + height >= DIM) ? DIM - 1 : y + height;

  for (int i = y_start; i < y_end; i++) {
    for (int j = x_start; j < x_end; j += 30) {
      // first we load three lines, our target line (v2) and the one above and
      // below. as we use tables that are larger by one, we don't really care
      // about oob writes, we will just need to mask out at the end.
      __m256i v1 = M256I_LOADU (i - 1, j - 1);
      __m256i v2 = M256I_LOADU (i, j - 1);
      __m256i v3 = M256I_LOADU (i + 1, j - 1);

      // we start by accumulating vertical neighbors.
      // now an entry of the vector represent the number of alive cells on the
      // three lines at this index.
      __m256i lines_sum = _mm256_add_epi8 (v1, v2);
      lines_sum         = _mm256_add_epi8 (lines_sum, v3);

      // now we will add left and right alive cell count by adding lines with
      // lines shifted respectively left and right
      __m256i lines_sum_left_shift  = shift_bytes_left (lines_sum);
      __m256i lines_sum_right_shift = shift_bytes_right (lines_sum);
      lines_sum = _mm256_add_epi8 (lines_sum, lines_sum_left_shift);
      lines_sum = _mm256_add_epi8 (lines_sum, lines_sum_right_shift);

      // then we need to compute a mask to remove from the count the value of
      // the cell we are working on to do so we create a mask of alive cells in
      // v2 so we know on which vector entries we need to substract 1
      __m256i neighbor_count = _mm256_sub_epi8 (lines_sum, v2);

      __attribute__ ((aligned (32))) char temp[32];
      _mm256_store_si256 ((__m256i *)temp, neighbor_count);
      for (int k = 1; k < 31; k++) {
        int n             = temp[k];
        int me            = cur_table (i, j - 1 + k);
        const char new_me = (me & ((n == 2) | (n == 3))) | (!me & (n == 3));
        change |= (me ^ new_me);
        next_table (i, j - 1 + k) = new_me;
      }
    }
  }

  return change;
}

static inline __m256i _mm256_compute_neighbors (
    __m256i vec_top_shift_left, __m256i vec_cell_shift_left,
    __m256i vec_bot_shift_left, __m256i vec_top, __m256i vec_cell,
    __m256i vec_bot, __m256i vec_top_shift_right, __m256i vec_cell_shift_right,
    __m256i vec_bot_shift_right)
{
  __m256i vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_top_shift_left, vec_cell_shift_left);
  vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_cell_line_neigh_count, vec_bot_shift_left);
  vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_cell_line_neigh_count, vec_top);
  vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_cell_line_neigh_count, vec_cell);
  vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_cell_line_neigh_count, vec_bot);
  vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_cell_line_neigh_count, vec_top_shift_right);
  vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_cell_line_neigh_count, vec_cell_shift_right);
  vec_cell_line_neigh_count =
      _mm256_add_epi8 (vec_cell_line_neigh_count, vec_bot_shift_right);

  // we substract ourself from this count
  vec_cell_line_neigh_count =
      _mm256_sub_epi8 (vec_cell_line_neigh_count, vec_cell);
  return vec_cell_line_neigh_count;
}
static inline __m256i
_mm256_compute_cells (__m256i vec_cell_line_neigh_count, __m256i vec_cell,
                      __m256i only_threes, __m256i only_twos, __m256i only_ones,
                      __m256i only_zeros)
{
  __m256i three_neighbors =
      _mm256_cmpeq_epi8 (vec_cell_line_neigh_count, only_threes);

  __m256i two_neighbors =
      _mm256_cmpeq_epi8 (vec_cell_line_neigh_count, only_twos);

  __m256i alive_mask = _mm256_cmpgt_epi8 (vec_cell, only_zeros);

  __m256i two_neighbors_and_alive =
      _mm256_and_si256 (two_neighbors, alive_mask);

  __m256i next_alive_mask =
      _mm256_or_si256 (three_neighbors, two_neighbors_and_alive);

  __m256i next_alive = _mm256_and_si256 (next_alive_mask, only_ones);
  return next_alive;
}

static inline char compute_from_vects (
    __m256i vec_top_shift_left, __m256i vec_cell_shift_left,
    __m256i vec_bot_shift_left, __m256i vec_top, __m256i vec_cell,
    __m256i vec_bot, __m256i vec_top_shift_right, __m256i vec_cell_shift_right,
    __m256i vec_bot_shift_right, int i, int j, __m256i only_threes,
    __m256i only_twos, __m256i only_ones, __m256i only_zeros)
{
  // then we compute the neighbor count
  __m256i vec_cell_line_neigh_count = _mm256_compute_neighbors (
      vec_top_shift_left, vec_cell_shift_left, vec_bot_shift_left, vec_top,
      vec_cell, vec_bot, vec_top_shift_right, vec_cell_shift_right,
      vec_bot_shift_right);
  // we can now apply rules
  __m256i next_alive =
      _mm256_compute_cells (vec_cell_line_neigh_count, vec_cell, only_threes,
                            only_twos, only_ones, only_zeros);
  // store the result
  _mm256_storeu_si256 ((__m256i *)table_cell (_alternate_table, i, j),
                       next_alive);

  // and finally compute for changes
  __m256i diff = _mm256_xor_si256 (vec_cell, next_alive);
  return !_mm256_testz_si256 (diff, diff);
}

int life_do_tile_avx2 (const int x, const int y, const int width,
                       const int height)
{
  if (x < 32 || x + width >= DIM - 33) {
    return life_do_tile_opt (x, y, width, height);
  }
  char change = 0;

  int x_start = (x == 0) ? 1 : x;
  int x_end   = (x + width >= DIM) ? DIM - 1 : x + width;
  int y_start = (y == 0) ? 1 : y;
  int y_end   = (y + height >= DIM) ? DIM - 1 : y + height;

  // some constants we're gonna use
  __m256i only_threes = _mm256_set1_epi8 (3);
  __m256i only_twos   = _mm256_set1_epi8 (2);
  __m256i only_ones   = _mm256_set1_epi8 (1);
  __m256i only_zeros  = _mm256_setzero_si256 ();

  for (int i = y_start; i < y_end; i++) {
    for (int j = x_start; j < x_end; j += 32) {
      __m256i vec_top_shift_left  = M256I_LOADU (i - 1, j - 1);
      __m256i vec_cell_shift_left = M256I_LOADU (i, j - 1);
      __m256i vec_bot_shift_left  = M256I_LOADU (i + 1, j - 1);

      __m256i vec_top  = M256I_LOADU (i - 1, j);
      __m256i vec_cell = M256I_LOADU (i, j);
      __m256i vec_bot  = M256I_LOADU (i + 1, j);

      __m256i vec_top_shift_right  = M256I_LOADU (i - 1, j + 1);
      __m256i vec_cell_shift_right = M256I_LOADU (i, j + 1);
      __m256i vec_bot_shift_right  = M256I_LOADU (i + 1, j + 1);

      change |= compute_from_vects (
          vec_top_shift_left, vec_cell_shift_left, vec_bot_shift_left, vec_top,
          vec_cell, vec_bot, vec_top_shift_right, vec_cell_shift_right,
          vec_bot_shift_right, i, j, only_threes, only_twos, only_ones,
          only_zeros);
    }
  }
  return change;
}
int life_do_tile_avx2_lessload (const int x, const int y, const int width,
                                const int height)
{
  if (x < 32 || x + width >= DIM - 33) {
    return life_do_tile_opt (x, y, width, height);
  }
  char change = 0;
  int x_start = (x == 0) ? 1 : x;
  int x_end   = (x + width >= DIM) ? DIM - 1 : x + width;
  int y_start = (y == 0) ? 1 : y;
  int y_end   = (y + height >= DIM) ? DIM - 1 : y + height;

  __m256i only_threes = _mm256_set1_epi8 (3);
  __m256i only_twos   = _mm256_set1_epi8 (2);
  __m256i only_ones   = _mm256_set1_epi8 (1);
  __m256i only_zeros  = _mm256_setzero_si256 ();

  for (int j = x_start; j < x_end; j += 32) {
    // reset i at the beginning of each outer loop iteration
    for (int i = y_start; i < y_end; i++) {
      // load all 9 vectors using the macro
      __m256i vec_top_shift_left  = M256I_LOADU (i - 1, j - 1);
      __m256i vec_cell_shift_left = M256I_LOADU (i, j - 1);
      __m256i vec_bot_shift_left  = M256I_LOADU (i + 1, j - 1);

      __m256i vec_top  = M256I_LOADU (i - 1, j);
      __m256i vec_cell = M256I_LOADU (i, j);
      __m256i vec_bot  = M256I_LOADU (i + 1, j);

      __m256i vec_top_shift_right  = M256I_LOADU (i - 1, j + 1);
      __m256i vec_cell_shift_right = M256I_LOADU (i, j + 1);
      __m256i vec_bot_shift_right  = M256I_LOADU (i + 1, j + 1);

      change |= compute_from_vects (
          vec_top_shift_left, vec_cell_shift_left, vec_bot_shift_left, vec_top,
          vec_cell, vec_bot, vec_top_shift_right, vec_cell_shift_right,
          vec_bot_shift_right, i, j, only_threes, only_twos, only_ones,
          only_zeros);
    }
  }
  return change;
}
#endif
#if __AVX512__ == 1
static inline __m512i _mm512_compute_neighbors (
    __m512i vec_top_shift_left, __m512i vec_cell_shift_left,
    __m512i vec_bot_shift_left, __m512i vec_top, __m512i vec_cell,
    __m512i vec_bot, __m512i vec_top_shift_right, __m512i vec_cell_shift_right,
    __m512i vec_bot_shift_right)
{
  __m512i vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_top_shift_left, vec_cell_shift_left);
  vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_cell_line_neigh_count, vec_bot_shift_left);
  vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_cell_line_neigh_count, vec_top);
  vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_cell_line_neigh_count, vec_cell);
  vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_cell_line_neigh_count, vec_bot);
  vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_cell_line_neigh_count, vec_top_shift_right);
  vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_cell_line_neigh_count, vec_cell_shift_right);
  vec_cell_line_neigh_count =
      _mm512_add_epi8 (vec_cell_line_neigh_count, vec_bot_shift_right);

  // we substract ourself from this count
  vec_cell_line_neigh_count =
      _mm512_sub_epi8 (vec_cell_line_neigh_count, vec_cell);
  return vec_cell_line_neigh_count;
}

static inline __m512i
_mm512_compute_cells (__m512i vec_cell_line_neigh_count, __m512i vec_cell,
                      __m512i only_threes, __m512i only_twos, __m512i only_ones,
                      __m512i only_zeros)
{
  __mmask64 three_neighbors_mask =
      _mm512_cmpeq_epi8_mask (vec_cell_line_neigh_count, only_threes);

  __mmask64 two_neighbors_mask =
      _mm512_cmpeq_epi8_mask (vec_cell_line_neigh_count, only_twos);

  __mmask64 alive_mask = _mm512_cmpgt_epi8_mask (vec_cell, only_zeros);

  __mmask64 two_neighbors_and_alive_mask = two_neighbors_mask & alive_mask;

  __mmask64 next_alive_mask =
      three_neighbors_mask | two_neighbors_and_alive_mask;

  __m512i next_alive =
      _mm512_mask_blend_epi8 (next_alive_mask, only_zeros, only_ones);
  return next_alive;
}

static inline char _mm512_compute_from_vects (
    __m512i vec_top_shift_left, __m512i vec_cell_shift_left,
    __m512i vec_bot_shift_left, __m512i vec_top, __m512i vec_cell,
    __m512i vec_bot, __m512i vec_top_shift_right, __m512i vec_cell_shift_right,
    __m512i vec_bot_shift_right, int i, int j, __m512i only_threes,
    __m512i only_twos, __m512i only_ones, __m512i only_zeros)
{
  // then we compute the neighbor count
  __m512i vec_cell_line_neigh_count = _mm512_compute_neighbors (
      vec_top_shift_left, vec_cell_shift_left, vec_bot_shift_left, vec_top,
      vec_cell, vec_bot, vec_top_shift_right, vec_cell_shift_right,
      vec_bot_shift_right);
  // we can now apply rules
  __m512i next_alive =
      _mm512_compute_cells (vec_cell_line_neigh_count, vec_cell, only_threes,
                            only_twos, only_ones, only_zeros);
  // store the result
  _mm512_storeu_si512 ((__m512i *)table_cell (_alternate_table, i, j),
                       next_alive);

  // and finally compute for changes
  __mmask64 diff_mask = _mm512_cmpneq_epi8_mask (vec_cell, next_alive);

  return (diff_mask != 0);
}

#define M512I_LOADU(y, x)                                                      \
  _mm512_loadu_si512 ((const __m512i *)table_cell (_table, (y), (x)))

int life_do_tile_avx512 (const int x, const int y, const int width,
                         const int height)
{
  if (x < 64 || x + width >= DIM - 65) {
    return life_do_tile_opt (x, y, width, height);
  }
  char change = 0;

  int x_start = (x == 0) ? 1 : x;
  int x_end   = (x + width >= DIM) ? DIM - 1 : x + width;
  int y_start = (y == 0) ? 1 : y;
  int y_end   = (y + height >= DIM) ? DIM - 1 : y + height;

  // some constants we're gonna use
  __m512i only_threes = _mm512_set1_epi8 (3);
  __m512i only_twos   = _mm512_set1_epi8 (2);
  __m512i only_ones   = _mm512_set1_epi8 (1);
  __m512i only_zeros  = _mm512_setzero_si512 ();

  for (int i = y_start; i < y_end; i++) {
    for (int j = x_start; j < x_end; j += 64) {
      __m512i vec_top_shift_left  = M512I_LOADU (i - 1, j - 1);
      __m512i vec_cell_shift_left = M512I_LOADU (i, j - 1);
      __m512i vec_bot_shift_left  = M512I_LOADU (i + 1, j - 1);

      __m512i vec_top  = M512I_LOADU (i - 1, j);
      __m512i vec_cell = M512I_LOADU (i, j);
      __m512i vec_bot  = M512I_LOADU (i + 1, j);

      __m512i vec_top_shift_right  = M512I_LOADU (i - 1, j + 1);
      __m512i vec_cell_shift_right = M512I_LOADU (i, j + 1);
      __m512i vec_bot_shift_right  = M512I_LOADU (i + 1, j + 1);

      change |= _mm512_compute_from_vects (
          vec_top_shift_left, vec_cell_shift_left, vec_bot_shift_left, vec_top,
          vec_cell, vec_bot, vec_top_shift_right, vec_cell_shift_right,
          vec_bot_shift_right, i, j, only_threes, only_twos, only_ones,
          only_zeros);
    }
  }
  return change;
}
int life_do_tile_avx_balanced (const int x, const int y, const int width,
                               const int height)
{
  int tid = omp_get_thread_num ();
  if (tid % 24 == 0) {
    return life_do_tile_avx512 (x, y, width, height);
  } else {
    return life_do_tile_avx2 (x, y, width, height);
  }
}

#endif
#endif
*/
///////////////////////////// Do tile optimized
int life_do_tile_opt (const int x, const int y, const int width,
                      const int height)
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

      // pretty sure this could run faster but it doesn't for now. keeping it
      // for later uint32_t top = *(uint32_t*)(cur_table(i-1, j-1)) &
      // 0x00FFFFFF; uint32_t mid = *(uint32_t*)(cur_table(i,   j-1)) &
      // 0x00FFFFFF; uint32_t bot = *(uint32_t*)(cur_table(i+1, j-1)) &
      // 0x00FFFFFF;

      // uint32_t neighborhood = (top << 16) | (mid << 8) | bot;

      // //neighborhood &= ~(1 << 8);

      // int n = __builtin_popcount(neighborhood);

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

unsigned life_compute_lazy_ompfor (unsigned nb_iter)
{
  unsigned res = 0;

  for (int it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
#pragma omp parallel for reduction(| : change) collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += TILE_H) {
      for (int x = 0; x < DIM; x += TILE_W) {
        unsigned local_change = 0;
        unsigned tile_y       = y / TILE_H;
        unsigned tile_x       = x / TILE_W;
        // checking if we should recompute this tile or not
        if (cur_dirty (tile_y, tile_x) || next_dirty (tile_y, tile_x)) {
          // we need to keep track of per-tile changes
          local_change = do_tile (x, y, TILE_W, TILE_H);
          change |= local_change;

          if (local_change) {
            // setting them to 2 in order to avoid writing 0 on a unchanged tile
            // that has some changes in its neighborhood
            next_dirty (tile_y - 1, tile_x - 1) = 1;
            next_dirty (tile_y - 1, tile_x)     = 1;
            next_dirty (tile_y - 1, tile_x + 1) = 1;
            next_dirty (tile_y, tile_x - 1)     = 1;
            next_dirty (tile_y, tile_x) =
                1; // except for the one of the iteration
            next_dirty (tile_y, tile_x + 1)     = 1;
            next_dirty (tile_y + 1, tile_x - 1) = 1;
            next_dirty (tile_y + 1, tile_x)     = 1;
            next_dirty (tile_y + 1, tile_x + 1) = 1;
          }
        }
      }
    }

    if (!change)
      return it;
    swap_tables_w_dirty ();
  }

  return res;
}

unsigned life_compute_lazy (unsigned nb_iter)
{
  unsigned res = 0;

  for (int it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    for (int y = 0; y < DIM; y += TILE_H) {
      unsigned tile_y = y / TILE_H;
      for (int x = 0; x < DIM; x += TILE_W) {
        unsigned local_change = 0;
        unsigned tile_y       = y / TILE_H;
        unsigned tile_x       = x / TILE_W;
        // checking if we should recompute this tile or not
        if (cur_dirty (tile_y, tile_x)) {
          // we need to keep track of per-tile changes
          local_change = do_tile (x, y, TILE_W, TILE_H);
          change |= local_change;

          if (local_change) {
            // setting them to 2 in order to avoid writing 0 on a unchanged tile
            // that has some changes in its neighborhood
            next_dirty (tile_y - 1, tile_x - 1) = 2;
            next_dirty (tile_y - 1, tile_x)     = 2;
            next_dirty (tile_y - 1, tile_x + 1) = 2;
            next_dirty (tile_y, tile_x - 1)     = 2;
            next_dirty (tile_y, tile_x) =
                1; // except for the one of the iteration
            next_dirty (tile_y, tile_x + 1)     = 2;
            next_dirty (tile_y + 1, tile_x - 1) = 2;
            next_dirty (tile_y + 1, tile_x)     = 2;
            next_dirty (tile_y + 1, tile_x + 1) = 2;
          } else {
            if (next_dirty (tile_y, tile_x) !=
                2) { // checking if needs to be recomputed for a neighbor
              next_dirty (tile_y, tile_x) = 0;
              cur_dirty (tile_y, tile_x)  = 0;
            }
          }
        }
      }
    }
    if (!change)
      return it;
    swap_tables_w_dirty ();
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

///////////////////////////// First touch allocations
void life_ft (void)
{
#pragma omp parallel for schedule(runtime) collapse(2)
  for (int y = 0; y < DIM; y += TILE_H)
    for (int x = 0; x < DIM; x += TILE_W) {
      unsigned tile_y   = y / TILE_H;
      unsigned tile_x   = x / TILE_W;
      next_table (y, x) = cur_table (y, x) = 0;
      next_dirty (tile_y, tile_x) = cur_dirty (tile_y, tile_x) = 1;
    }
}



///////////////////////////// MPI
void life_init_mpi(){
  easypap_check_mpi();
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
}

static int rankTop(int rank)
{
    return rank * (DIM / size);
}

static unsigned int rankSize(int rank)
{
  return (rank != size - 1) ? (DIM / size) : ((DIM / size) + (DIM % size));
}

static void recv_lines (int ligne, int from){
MPI_Status status;
MPI_Recv (&cur_table (ligne, 0), DIM, MPI_UNSIGNED, from, 0, MPI_COMM_WORLD, &status);
}
static void send_lines (int ligne, int to){
MPI_Send (&cur_table (ligne, 0), DIM, MPI_UNSIGNED, to, 0, MPI_COMM_WORLD);
}

static void send_borders (void){
  if (rank % 2 == 0) {
  send_lines (rankTop (rank), rank != 0 ? (rank - 1) : (size - 1));
  recv_lines (rankTop ((rank + 1) % size), (rank + 1) % size);
  } else {
  recv_lines (rankTop ((rank + 1) % size), (rank + 1) % size);
  send_lines (rankTop (rank), rank != 0 ? (rank - 1) : (size - 1));
  }
}  

void life_refresh_img_mpi(){
  MPI_Status status;

  if (rank == 0) {
    for (int i = 1; i < size; i++) {
        unsigned otherRankTop = rankTop(i);
        unsigned otherRankSize = rankSize(i);
        
        MPI_Recv(&cur_table(otherRankTop, 0), 
                 DIM * otherRankSize, MPI_UNSIGNED, 
                 i, 0, MPI_COMM_WORLD, &status);
    }       
} 
else {
    unsigned myTop = rankTop(rank);
    unsigned mySize = rankSize(rank);
    
    MPI_Send(&cur_table(myTop, 0), 
             DIM * mySize, MPI_UNSIGNED, 
             0, 0, MPI_COMM_WORLD);
}
life_refresh_img();
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
