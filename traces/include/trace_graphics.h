#ifndef TRACE_GRAPHICS_IS_DEF
#define TRACE_GRAPHICS_IS_DEF


#include "trace_data.h"


void trace_graphics_init (unsigned w, unsigned h);
void trace_graphics_process_event (SDL_Event *event);
void trace_graphics_setview (int start_iteration, int end_iteration);
void trace_graphics_display_all (void);
void trace_graphics_clean ();

extern int use_thumbnails;
extern unsigned char brightness;
extern unsigned soft_rendering;

#endif
