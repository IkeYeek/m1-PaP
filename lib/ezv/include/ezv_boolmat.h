#ifndef EZV_MATHNN_H
#define EZV_MATHNN_H


struct _boolmat;
typedef struct _boolmat ezv_boolmat_t;

ezv_boolmat_t *ezv_boolmat_alloc (unsigned rows, unsigned cols);
void ezv_boolmat_free (ezv_boolmat_t *mat);

unsigned ezv_boolmat_get (ezv_boolmat_t *mat, unsigned row, unsigned col);
void ezv_boolmat_setval (ezv_boolmat_t *mat, unsigned row, unsigned col, unsigned v);
void ezv_boolmat_set (ezv_boolmat_t *mat, unsigned row, unsigned col);
void ezv_boolmat_clear (ezv_boolmat_t *mat, unsigned row, unsigned col);
unsigned ezv_boolmat_sum_row (ezv_boolmat_t *mat, unsigned row);

void ezv_boolmat_display (ezv_boolmat_t *mat);


#endif