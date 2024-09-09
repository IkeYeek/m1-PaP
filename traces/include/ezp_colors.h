#ifndef EZP_COLORS_H
#define EZP_COLORS_H


#include <stdint.h>

#define EZP_MAX_COLORS 14

void ezp_colors_init (void);

extern uint32_t ezp_cpu_colors[];
extern unsigned ezp_gpu_index[];


#endif