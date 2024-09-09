#ifndef EZV_SHADER_H
#define EZV_SHADER_H

#include "ezv_sdl_gl.h"

extern const char *ezv_prefix;

GLuint ezv_shader_create (const char *vertex_shader,
                          const char *geometry_shader,
                          const char *fragment_shader);
void ezv_shader_get_uniform_loc (GLuint program, const char *name,
                                 GLuint *location);
void ezv_shader_bind_uniform_buf (GLuint program, const char *name,
                                  GLuint blockbinding);

#endif
