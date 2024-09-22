#include "ezv_boolmat.h"
#include "error.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define pack_t uint64_t
#define bits_minus_one 63U
#define div_pack(n) ((n) >> 6)
#define mod_pack(n) ((n) & bits_minus_one)

struct _boolmat
{
  unsigned cols, rows;
  unsigned rcols;
  pack_t *data;
};

static void check_bounds (ezv_boolmat_t *mat, unsigned row, unsigned col)
{
  if (row >= mat->rows)
    exit_with_error (
        "row parameter (%d) is out of bound (matrix size: %d rows x %d cols)",
        row, mat->rows, mat->cols);
  if (col >= mat->cols)
    exit_with_error (
        "col parameter (%d) is out of bound (matrix size : %d rows x %d cols)",
        col, mat->rows, mat->cols);
}

ezv_boolmat_t *ezv_boolmat_alloc (unsigned rows, unsigned cols)
{
  ezv_boolmat_t *mat = malloc (sizeof (ezv_boolmat_t));

  mat->rows  = rows;
  mat->cols  = cols;
  mat->rcols = div_pack (cols + bits_minus_one);
  mat->data  = calloc (mat->rcols * rows, sizeof (pack_t));

  return mat;
}

void ezv_boolmat_free (ezv_boolmat_t *mat)
{
  free (mat->data);
  free (mat);
}

unsigned ezv_boolmat_get (ezv_boolmat_t *mat, unsigned row, unsigned col)
{
  check_bounds (mat, row, col);

  uint64_t v = mat->data[mat->rcols * row + div_pack (col)];

  return (v >> mod_pack (col)) & 1U;
}

void ezv_boolmat_setval (ezv_boolmat_t *mat, unsigned row, unsigned col,
                         unsigned v)
{
  check_bounds (mat, row, col);

  const unsigned div_c = div_pack (col);
  const unsigned mod_c = mod_pack (col);
  uint64_t mask        = ~((uint64_t)1 << mod_c);
  uint64_t vv          = (uint64_t)v << mod_c;

  mat->data[mat->rcols * row + div_c] =
      (mat->data[mat->rcols * row + div_c] & mask) | vv;
}

void ezv_boolmat_set (ezv_boolmat_t *mat, unsigned row, unsigned col)
{
  ezv_boolmat_setval (mat, row, col, 1);
}

void ezv_boolmat_clear (ezv_boolmat_t *mat, unsigned row, unsigned col)
{
  ezv_boolmat_setval (mat, row, col, 0);
}

// Fast method for computing the number of bits set in a 64-bit integer
static unsigned nb_bits (uint64_t v)
{
  v = v - ((v >> 1) & 0x5555555555555555);
  v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333);
  return (((v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56;
}

unsigned ezv_boolmat_sum_row (ezv_boolmat_t *mat, unsigned row)
{
  check_bounds (mat, row, 0);

  unsigned sum = 0;
  for (int c = 0; c < mat->rcols; c++)
    sum += nb_bits (mat->data[mat->rcols * row + c]);

  return sum;
}

void ezv_boolmat_display (ezv_boolmat_t *mat)
{
  for (int r = 0; r < mat->rows; r++) {
    for (int c = 0; c < mat->cols; c++)
      printf ("%c ", ezv_boolmat_get (mat, r, c) + '0');
    printf ("(sum = %d)\n", ezv_boolmat_sum_row (mat, r));
  }
}
