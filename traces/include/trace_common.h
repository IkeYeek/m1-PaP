#ifndef TRACE_COMMON_IS_DEF
#define TRACE_COMMON_IS_DEF


#define TRACE_BEGIN_ITER   0x101
#define TRACE_BEGIN_TILE   0x102
#define TRACE_END_TILE     0x103
#define TRACE_NB_THREADS   0x104
#define TRACE_NB_ITER      0x105
#define TRACE_DIM          0x106
#define TRACE_END_ITER     0x107
#define TRACE_LABEL        0x108
#define TRACE_TASKID_COUNT 0x109
#define TRACE_TASKID       0x10A
#define TRACE_FIRST_ITER   0x10B
#define TRACE_DO_CACHE     0x10C
#define TRACE_TILE         0x10D
#define TRACE_MESHFILE     0x10E
#define TRACE_PALETTE      0x10F
#define TRACE_PATCH        0x110
#define TRACE_TILE_EXT     0x111
#define TRACE_PATCH_EXT    0x112
#define TRACE_TILE_MIN     0x113
#define TRACE_PATCH_MIN    0x114

typedef enum {
    TASK_TYPE_COMPUTE,
    TASK_TYPE_WRITE,
    TASK_TYPE_READ
} task_type_t;

#define INT_COMBINE(low,high) ((unsigned long)(low) | ((unsigned long)(high) << 32))
#define INT_EXTRACT_HIGH(v)   ((unsigned long)(v) >> 32)
#define INT_EXTRACT_LOW(v)   ((unsigned long)(v) & ((1UL << 32) - 1))


#endif
