#ifndef EZP_CTX_H
#define EZP_CTX_H


#include "ezv.h"

extern ezv_ctx_t ctx[];
extern unsigned nb_ctx;

void ezp_ctx_init (void);
int ezp_ctx_create (ezv_ctx_type_t ctx_type);

void ezp_ctx_ithud_init (int show);
void ezp_ctx_ithud_toggle (void);
void ezp_ctx_ithud_set (unsigned iter);

void ezp_ctx_coord_next (ezv_ctx_type_t ctx_type, unsigned ctx_no, int *xwin, int *ywin);


#endif
