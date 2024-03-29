#ifndef MESH3D_IS_DEF
#define MESH3D_IS_DEF

#include <SDL2/SDL.h>

#include "mesh3d_obj.h"
#include "mesh3d_palette.h"

#define MESH3D_ENABLE_PICKING 1
#define MESH3D_ENABLE_VSYNC 2
#define MESH3D_ENABLE_HUD 4
#define MESH3D_ENABLE_CLIPPING 8

struct mesh3d_ctx_s;
typedef struct mesh3d_ctx_s *mesh3d_ctx_t;

void mesh3d_init (const char *prefix);
void mesh3d_load_opengl (void);

mesh3d_ctx_t mesh3d_ctx_create (const char *win_title, int x, int y, int w, int h,
                        int flags);
SDL_Window *mesh3d_sdl_window (mesh3d_ctx_t ctx);
SDL_GLContext mesh3d_glcontext (mesh3d_ctx_t ctx);
void mesh3d_switch_to_context (mesh3d_ctx_t ctx);
void mesh3d_ctx_destroy (mesh3d_ctx_t ctx);

void mesh3d_set_mesh (mesh3d_ctx_t ctx, mesh3d_obj_t *mesh);

void mesh3d_use_cpu_colors (mesh3d_ctx_t ctx);
void mesh3d_reset_cpu_colors (mesh3d_ctx_t ctx);
void mesh3d_set_cpu_color (mesh3d_ctx_t ctx, unsigned first_cell,
                           unsigned num_cells, unsigned color);

void mesh3d_configure_data_colors_predefined (mesh3d_ctx_t ctx,
                                              mesh3d_palette_name_t palette);
void mesh3d_configure_data_colors (mesh3d_ctx_t ctx, float *data,
                                   unsigned size);
void mesh3d_set_data_colors (mesh3d_ctx_t ctx, float *values);
void mesh3d_switch_data_color_buffer (mesh3d_ctx_t ctx);
void mesh3d_get_sharable_buffer_ids (mesh3d_ctx_t ctx, int buffer_ids[]);
void mesh3d_set_data_brightness (mesh3d_ctx_t ctx, float brightness);

void mesh3d_process_event (mesh3d_ctx_t ctx[], unsigned nb_ctx,
                           SDL_Event *event, int *refresh, int *pick);

void mesh3d_reset_view (mesh3d_ctx_t ctx[], unsigned nb_ctx);

int mesh3d_ctx_is_in_focus (mesh3d_ctx_t ctx);
int mesh3d_perform_picking (mesh3d_ctx_t ctx[], unsigned nb_ctx);

void mesh3d_hud_init (mesh3d_ctx_t ctx);
int mesh3d_hud_alloc (mesh3d_ctx_t ctx);
void mesh3d_hud_free (mesh3d_ctx_t ctx, int hud);
void mesh3d_hud_toggle (mesh3d_ctx_t ctx, int hud);
void mesh3d_hud_on (mesh3d_ctx_t ctx, int hud);
void mesh3d_hud_off (mesh3d_ctx_t ctx, int hud);

void mesh3d_hud_set (mesh3d_ctx_t ctx, int hud, char *format, ...);


void mesh3d_render (mesh3d_ctx_t ctx[], unsigned nb_ctx);


#endif
