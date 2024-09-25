#ifndef HOOKS_IS_DEF
#define HOOKS_IS_DEF

#include <omp.h>

typedef void (*void_func_t) (void);
typedef unsigned (*int_func_t) (unsigned);
typedef void (*draw_func_t) (char *);
typedef int (*tile_func_t) (int, int, int, int);
typedef int (*patch_func_t) (int, int);
typedef void (*cuda_kernel_func_t)(unsigned *, unsigned *, unsigned);
typedef void (*cuda_kernel_finish_func_t)(unsigned);
typedef void (*debug_1d_t) (int);
typedef void (*debug_2d_t) (int, int);

extern draw_func_t the_config;
extern void_func_t the_init;
extern void_func_t the_first_touch;
extern draw_func_t the_draw;
extern void_func_t the_finalize;
extern int_func_t the_compute;
extern void_func_t the_tile_check;
extern cuda_kernel_func_t the_cuda_kernel;
extern cuda_kernel_finish_func_t the_cuda_kernel_post;
extern debug_1d_t the_1d_debug;
extern debug_2d_t the_2d_debug;
extern debug_1d_t the_1d_overlay;
extern debug_2d_t the_2d_overlay;
extern void_func_t the_send_data;

void *bind_it (const char *kernel, const char *s, const char *variant, int print_error);
void *hooks_find_symbol (char *symbol);
void hooks_establish_bindings (int silent);

// Call function ${kernel}_draw_${suffix}, or default_func if symbol not found
void hooks_draw_helper (char *suffix, void_func_t default_func);

int hooks_refresh_img (void);

// Call appropriate do_tile_${suffix} function, with calls to monitoring start/end
int do_tile_id (int x, int y, int width, int height, int who);

#define do_tile_implicit(x,y,w,h) do_tile_id(x,y,w,h,omp_get_thread_num())

// The two macros implement the switch between do_tile_implicit and do_tile_id
#define _macro(_1,_2,_3,_4,_5,NAME,...) NAME
#define do_tile(...) _macro(__VA_ARGS__,do_tile_id,do_tile_implicit)(__VA_ARGS__)

// Call appropriate do_patch_${suffix} function, with calls to monitoring start/end
int do_patch_id (int patch, int who);

#define do_patch_implicit(p) do_patch_id((p),omp_get_thread_num())

#define _macrop(_1,_2,NAME,...) NAME
#define do_patch(...) _macrop(__VA_ARGS__,do_patch_id,do_patch_implicit)(__VA_ARGS__)


#endif
