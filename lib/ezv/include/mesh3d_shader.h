#ifndef MESH3D_SHADER_IS_DEF
#define MESH3D_SHADER_IS_DEF

#include "mesh3d_sdl_gl.h"

extern const char *mesh3d_prefix;

GLuint mesh3d_shader_create (const char *vertex_shader,
                             const char *geometry_shader,
                             const char *fragment_shader);
void mesh3d_shader_get_uniform_loc (GLuint program, const char *name,
                                    GLuint *location);
void mesh3d_shader_bind_uniform_buf (GLuint program, const char *name,
                                     GLuint blockbinding);

#endif
