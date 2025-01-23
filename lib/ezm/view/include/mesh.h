#ifndef MESH_IS_DEF
#define MESH_IS_DEF

#include <SDL.h>

void mesh_init (int x, int y);
void mesh_load (char *filename);
void mesh_process_event (SDL_Event *event);
void mesh_set_cell_color (int triangle, int color, int highlight);
void mesh_reset_cell_colors (void);
void mesh_redraw (void);
void mesh_clean (void);

#endif
