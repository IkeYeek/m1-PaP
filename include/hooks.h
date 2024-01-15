#ifndef HOOKS_IS_DEF
#define HOOKS_IS_DEF

typedef void (*void_func_t) (void);
typedef unsigned (*int_func_t) (unsigned);
typedef void (*draw_func_t) (char *);
typedef int (*tile_func_t) (int, int, int, int);
typedef void (*cuda_kernel_func_t)(unsigned *, unsigned *, unsigned);
typedef void (*cuda_kernel_finish_func_t)(unsigned);

extern draw_func_t the_config;
extern void_func_t the_init;
extern void_func_t the_first_touch;
extern draw_func_t the_draw;
extern void_func_t the_finalize;
extern int_func_t the_compute;
extern void_func_t the_refresh_img;
extern void_func_t the_tile_check;
extern cuda_kernel_func_t the_cuda_kernel;
extern cuda_kernel_finish_func_t the_cuda_kernel_finish;

void *bind_it (char *kernel, char *s, char *variant, int print_error);
void *hooks_find_symbol (char *symbol);
void hooks_establish_bindings (int silent);

// Call function ${kernel}_draw_${suffix}, or default_func if symbol not found
void hooks_draw_helper (char *suffix, void_func_t default_func);

// Call appropriate do_tile_${suffix} function, with calls to monitoring start/end
int do_tile_id (int x, int y, int width, int height, int who);

#define do_tile_implicit(x,y,w,h) do_tile_id(x,y,w,h,omp_get_thread_num())

// The two macros implement the switch between do_tile_implicit and do_tile_id
#define _macro(_1,_2,_3,_4,_5,NAME,...) NAME
#define do_tile(...) _macro(__VA_ARGS__,do_tile_id,do_tile_implicit)(__VA_ARGS__)

#endif
