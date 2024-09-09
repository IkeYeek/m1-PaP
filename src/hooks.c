#include "hooks.h"
#include "debug.h"
#include "error.h"
#include "global.h"
#include "gpu.h"
#include "mesh_data.h"
#include "monitoring.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#define DLSYM_FLAG RTLD_SELF
#else
#define DLSYM_FLAG NULL
#endif

draw_func_t the_config                           = NULL;
void_func_t the_init                             = NULL;
void_func_t the_first_touch                      = NULL;
draw_func_t the_draw                             = NULL;
void_func_t the_finalize                         = NULL;
int_func_t the_compute                           = NULL;
void_func_t the_tile_check                       = NULL;
cuda_kernel_func_t the_cuda_kernel               = NULL;
cuda_kernel_finish_func_t the_cuda_kernel_finish = NULL;
debug_1d_t the_1d_debug                          = NULL;
debug_2d_t the_2d_debug                          = NULL;
debug_1d_t the_1d_overlay                        = NULL;
debug_2d_t the_2d_overlay                        = NULL;
void_func_t the_send_data                        = NULL;

static void_func_t the_refresh_img = NULL;
static tile_func_t the_tile_func   = NULL;
static patch_func_t the_patch_func = NULL;

void *hooks_find_symbol (char *symbol)
{
  return dlsym (DLSYM_FLAG, symbol);
}

void *bind_it (char *kernel, char *s, char *variant, int print_error)
{
  char buffer[1024];
  void *fun = NULL;

  if (s == NULL)
    sprintf (buffer, "%s_%s", kernel, variant);
  else
    sprintf (buffer, "%s_%s_%s", kernel, s, variant);

  fun = hooks_find_symbol (buffer);
  if (fun != NULL)
    PRINT_DEBUG ('h', "Found [%s]\n", buffer);
  else {
    if (print_error == 2)
      exit_with_error ("Cannot resolve function [%s]", buffer);

    sprintf (buffer, "%s_%s", kernel, s);
    fun = hooks_find_symbol (buffer);

    if (fun != NULL)
      PRINT_DEBUG ('h', "Found [%s]\n", buffer);
    else if (print_error)
      exit_with_error ("Cannot resolve function [%s]", buffer);
  }
  return fun;
}

static int no_tile_func (int x, int y, int w, int h)
{
  if (easypap_mode == EASYPAP_MODE_2D_IMAGES)
    exit_with_error ("No appropriate do_tile function found");
  else
    exit_with_error ("do_tile cannot be used in 3D mesh mode");

  return -1;
}

static void *bind_tile (char *kernel)
{
  char buffer[1024];
  void *fun = NULL;

  // First try to obey user
  if (tile_name != NULL) {
    sprintf (buffer, "%s_do_tile_%s", kernel, tile_name);
    fun = hooks_find_symbol (buffer);
    if (fun != NULL) {
      PRINT_DEBUG ('h', "Found requested tiling func [%s]\n", buffer);
      return fun;
    }
    // requested tile_name didn't work
    exit_with_error ("Cannot resolve function [%s]\n", buffer);
  }

  // Try to explore EASYPAP_TILEPREF environment variable
  if (tile_name == NULL) {
    char flavor[128];
    char *env = getenv ("EASYPAP_TILEPREF");
    if (env != NULL) {
      int index_env = 0, index = 0;
      for (;;) {
        if (env[index_env] == ':' ||
            env[index_env] == '\0') { // end of flavor name
          flavor[index] = '\0';
          sprintf (buffer, "%s_do_tile_%s", kernel, flavor);
          fun = hooks_find_symbol (buffer);
          if (fun != NULL) {
            PRINT_DEBUG ('h', "Found preferred tiling func [%s]\n", buffer);
            tile_name = malloc (strlen (flavor) + 1);
            strcpy (tile_name, flavor);
            return fun;
          }
          // flavor not found
          if (env[index_env] == '\0')
            break;
          index_env++;
          index = 0;
        }
        // regular character
        flavor[index++] = env[index_env++];
      }
    }
  }

  // Well, try default do_tile function
  sprintf (buffer, "%s_do_tile_default", kernel);
  fun = hooks_find_symbol (buffer);
  if (fun != NULL) {
    PRINT_DEBUG ('h', "Found [%s]\n", buffer);
    tile_name = "default";
    return fun;
  }

  // No tile function found
  tile_name = "none";
  return no_tile_func;
}

static int no_patch_func (int c, int n)
{
  if (easypap_mode == EASYPAP_MODE_3D_MESHES)
    exit_with_error ("No appropriate do_patch function found");
  else
    exit_with_error ("do_patch cannot be used in 2D image mode");

  return -1;
}

static void *bind_patch (char *kernel)
{
  char buffer[1024];
  void *fun = NULL;

  // First try to obey user
  if (tile_name != NULL) {
    sprintf (buffer, "%s_do_patch_%s", kernel, tile_name);
    fun = hooks_find_symbol (buffer);
    if (fun != NULL) {
      PRINT_DEBUG ('h', "Found requested patch func [%s]\n", buffer);
      return fun;
    }
    // requested tile_name didn't work
    exit_with_error ("Cannot resolve function [%s]\n", buffer);
  }

  // Try to explore EASYPAP_TILEPREF environment variable
  if (tile_name == NULL) {
    char flavor[128];
    char *env = getenv ("EASYPAP_TILEPREF");
    if (env != NULL) {
      int index_env = 0, index = 0;
      for (;;) {
        if (env[index_env] == ':' ||
            env[index_env] == '\0') { // end of flavor name
          flavor[index] = '\0';
          sprintf (buffer, "%s_do_patch_%s", kernel, flavor);
          fun = hooks_find_symbol (buffer);
          if (fun != NULL) {
            PRINT_DEBUG ('h', "Found preferred patch func [%s]\n", buffer);
            tile_name = malloc (strlen (flavor) + 1);
            strcpy (tile_name, flavor);
            return fun;
          }
          // flavor not found
          if (env[index_env] == '\0')
            break;
          index_env++;
          index = 0;
        }
        // regular character
        flavor[index++] = env[index_env++];
      }
    }
  }

  // Well, try default do_tile function
  sprintf (buffer, "%s_do_patch_default", kernel);
  fun = hooks_find_symbol (buffer);
  if (fun != NULL) {
    PRINT_DEBUG ('h', "Found [%s]\n", buffer);
    tile_name = "default";
    return fun;
  }

  // No tile function found
  tile_name = "none";
  return no_patch_func;
}

void hooks_establish_bindings (int silent)
{
  if (gpu_used) {
    gpu_establish_bindings ();
  } else {
    the_compute     = bind_it (kernel_name, "compute", variant_name, 2);
    the_first_touch = bind_it (kernel_name, "ft", variant_name, do_first_touch);
  }

  the_config      = bind_it (kernel_name, "config", variant_name, 0);
  the_init        = bind_it (kernel_name, "init", variant_name, 0);
  the_draw        = bind_it (kernel_name, "draw", variant_name, 0);
  the_finalize    = bind_it (kernel_name, "finalize", variant_name, 0);
  the_refresh_img = bind_it (kernel_name, "refresh_img", variant_name, 0);

  if (easypap_mode == EASYPAP_MODE_2D_IMAGES) {
    the_tile_func  = bind_tile (kernel_name);
    the_tile_check = bind_it (kernel_name, "tile_check", tile_name, 0);
    the_patch_func = no_patch_func;
    the_2d_debug   = bind_it (kernel_name, "debug", variant_name, 0);
    the_2d_overlay = bind_it (kernel_name, "overlay", variant_name, 0);
  } else if (easypap_mode == EASYPAP_MODE_3D_MESHES) {
    the_patch_func = bind_patch (kernel_name);
    the_tile_check = bind_it (kernel_name, "patch_check", tile_name, 0);
    the_tile_func  = no_tile_func;
    the_1d_debug   = bind_it (kernel_name, "debug", variant_name, 0);
    the_1d_overlay = bind_it (kernel_name, "overlay", variant_name, 0);
  }

  if (!silent)
    PRINT_MASTER ("Using kernel [%s], variant [%s], tiling [%s]\n", kernel_name,
                  variant_name, tile_name);
}

void hooks_draw_helper (char *suffix, void_func_t default_func)
{
  char func_name[1024];
  void_func_t f = NULL;

  if (suffix == NULL)
    f = default_func;
  else {
    sprintf (func_name, "%s_draw_%s", kernel_name, suffix);
    f = hooks_find_symbol (func_name);

    if (f == NULL) {
      PRINT_DEBUG ('h', "Cannot resolve draw function: %s\n", func_name);
      f = default_func;
    }
  }

  f ();
}

int hooks_refresh_img (void)
{
  if (the_refresh_img) {
    the_refresh_img ();
    PRINT_DEBUG ('h', "refresh_img hook called\n");
    return 1;
  } else
    return 0;
}

int do_tile_id (int x, int y, int width, int height, int who)
{
  uint64_t clock = monitoring_start_tile (who);

  int r = the_tile_func (x, y, width, height);

  monitoring_end_tile (clock, x, y, width, height, who);

  return r;
}

int do_patch_id (int patch, int who)
{
  uint64_t clock = monitoring_start_tile (who);

  int r = the_patch_func (patch_start (patch), patch_end (patch));

  monitoring_end_tile (clock, patch_start (patch), 0, patch_size (patch), 0,
                       who);

  return r;
}