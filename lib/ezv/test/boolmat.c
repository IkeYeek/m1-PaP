#include <stdlib.h>
#include <stdio.h>

#include "ezv_boolmat.h"

int main (int argc, char *argv[])
{
  ezv_boolmat_t *m = ezv_boolmat_alloc (5, 67);

  for (int i = 0; i < 5; i++) {
    ezv_boolmat_set (m, i, i);
    ezv_boolmat_set (m, i, 33);
    ezv_boolmat_set (m, 2, 31 + i);
    ezv_boolmat_set (m, i, 66 - i);
  }

  ezv_boolmat_clear (m, 2, 33);

  ezv_boolmat_display (m);

  ezv_boolmat_free (m);
}
