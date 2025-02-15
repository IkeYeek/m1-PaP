
#ifndef GLOBAL_IS_DEF
#define GLOBAL_IS_DEF

extern unsigned do_display;
extern unsigned vsync;
extern unsigned soft_rendering;
extern unsigned refresh_rate;
extern unsigned do_first_touch;
extern int max_iter;
extern char *easypap_image_file;
extern char *easypap_mesh_file;
extern char *draw_param;
extern char *config_param;

extern unsigned gpu_used;
extern unsigned use_multiple_gpu;
extern unsigned easypap_mpirun;
extern unsigned easypap_gl_buffer_sharing;
extern unsigned picking_enabled;
extern unsigned use_scotch;

extern char *kernel_name, *variant_name, *tile_name;

typedef enum
{
  EASYPAP_MODE_UNDEFINED,
  EASYPAP_MODE_2D_IMAGES,
  EASYPAP_MODE_3D_MESHES
} easypap_mode_t;

extern easypap_mode_t easypap_mode;

#endif
