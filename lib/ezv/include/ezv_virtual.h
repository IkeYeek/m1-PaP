#ifndef EZV_VIRTUAL_H
#define EZV_VIRTUAL_H

struct ezv_ctx_s;
typedef struct ezv_ctx_s *ezv_ctx_t;

typedef struct
{
  void (*render) (ezv_ctx_t ctx);
  void (*reset_view) (ezv_ctx_t ctx[], unsigned nb_ctx);
  unsigned (*get_color_data_size) (ezv_ctx_t ctx);
  void (*activate_rgba_palette) (ezv_ctx_t ctx);
  void (*activate_data_palette) (ezv_ctx_t ctx);
  void (*get_shareable_buffer_ids) (ezv_ctx_t ctx, int buffer_ids[]);
  void (*set_data_brightness) (ezv_ctx_t ctx, float brightness);
  int (*do_1D_picking) (ezv_ctx_t ctx, int x, int y);
  void (*do_2D_picking) (ezv_ctx_t ctx, int mousex, int mousey, int *x, int *y);
  int (*zoom) (ezv_ctx_t ctx[], unsigned nb_ctx, unsigned shift_mod,
               unsigned in);
  int (*motion) (ezv_ctx_t ctx[], unsigned nb_ctx, int dx, int dy,
                 unsigned wheel);
  void (*move_zplane) (ezv_ctx_t ctx[], unsigned nb_ctx, float dz);
  void (*set_data_colors) (ezv_ctx_t ctx, void *values);
  unsigned (*get_linepitch) (ezv_ctx_t ctx);
} ezv_class_t;

// Private virtual methods
unsigned ezv_get_color_data_size (ezv_ctx_t ctx);
void ezv_activate_rgba_palette (ezv_ctx_t ctx);
void ezv_activate_data_palette (ezv_ctx_t ctx);
int ezv_do_1D_picking (ezv_ctx_t ctx, int mousex, int mousey);
void ezv_do_2D_picking (ezv_ctx_t ctx, int mousex, int mousey, int *x, int *y);
int ezv_zoom (ezv_ctx_t ctx[], unsigned nb_ctx, unsigned shift_mod,
              unsigned in);
int ezv_motion (ezv_ctx_t ctx[], unsigned nb_ctx, int dx, int dy,
                unsigned wheel);
void ezv_move_zplane (ezv_ctx_t ctx[], unsigned nb_ctx, float dz);
unsigned ezv_get_linepitch (ezv_ctx_t ctx);

#endif
