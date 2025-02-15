#ifndef EZM_PACK_H
#define EZM_PACK_H


#define INT_COMBINE(low,high) ((uint64_t)(low) | ((uint64_t)(high) << 32))
#define INT_EXTRACT_HIGH(v)   ((uint64_t)(v) >> 32)
#define INT_EXTRACT_LOW(v)    ((uint64_t)(v) & ((1ULL << 32) - 1))


#endif