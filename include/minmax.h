#ifndef MINMAX_IS_DEF
#define MINMAX_IS_DEF


#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define MAX(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define ROUND_TO_MULTIPLE(n, r) (((n) + (r) - 1U) & ~((r) - 1U))


#endif
