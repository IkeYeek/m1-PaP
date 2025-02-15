#ifndef EZP_ALLOC_H
#define EZP_ALLOC_H

#include <stdlib.h>

void *ezp_alloc (size_t size);

void ezp_free (void *ptr, size_t size);

#endif
