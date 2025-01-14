#ifndef TRACE_COLORS_H
#define TRACE_COLORS_H


#include <stdint.h>

extern unsigned TRACE_MAX_COLORS;

void trace_colors_init (unsigned npus);

uint32_t trace_cpu_color (int pu);


#endif