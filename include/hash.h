#ifndef HASH_IS_DEF
#define HASH_IS_DEF

#include <sys/types.h>

void build_hash_and_store_to_file (void *buffer, size_t len,
                                   const char *filename);

#endif
