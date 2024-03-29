#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "error.h"
#include "mesh3d_shader.h"

const char *mesh3d_prefix = NULL;

static size_t file_size (const char *filename)
{
  struct stat sb;

  if (stat (filename, &sb) < 0)
    exit_with_error ("Cannot access \"%s\" file (%s)", filename,
                     strerror (errno));
  return sb.st_size;
}

static char *file_load (const char *file)
{
  FILE *f;
  char *b;
  size_t s;
  size_t r;

  char filename[1024];
  sprintf (filename, "%s/shaders/%s", (mesh3d_prefix ? mesh3d_prefix : "."),
           file);

  s = file_size (filename);
  b = malloc (s + 1);
  if (!b)
    exit_with_error ("Malloc failed (%s)", strerror (errno));

  f = fopen (filename, "r");
  if (f == NULL)
    exit_with_error ("Cannot open \"%s\" file (%s)", filename,
                     strerror (errno));

  r = fread (b, s, 1, f);
  if (r != 1)
    exit_with_error ("fread failed (%s)", strerror (errno));

  b[s] = '\0';

  return b;
}

static void compile_shader (unsigned int shader)
{
  glCompileShader (shader);
  // check for shader compile errors
  GLint success;
  glGetShaderiv (shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    GLint length;
    char *log;

    glGetShaderiv (shader, GL_INFO_LOG_LENGTH, &length);
    log = malloc (length);
    glGetShaderInfoLog (shader, length, &success, log);
    check (success, "Failed to retrieve shader compiler log");
    exit_with_error ("Shader Compilation Error: %s\n", log);
  }
}

static void link_program (unsigned int shader_program)
{
  glLinkProgram (shader_program);

  GLint success;
  glGetProgramiv (shader_program, GL_LINK_STATUS, &success);
  if (!success) {
    GLint length;
    char *log;

    glGetProgramiv (shader_program, GL_INFO_LOG_LENGTH, &length);
    log = malloc (length);
    glGetProgramInfoLog (shader_program, length, &success, log);
    check (success, "Failed to retrieve shader compiler log");
    exit_with_error ("Shader Linking Error: %s\n", log);
  }
}

GLuint mesh3d_shader_create (const char *vertex_shader,
                             const char *geometry_shader,
                             const char *fragment_shader)
{
  char *source;
  // vertex shader
  GLuint vertexShader = glCreateShader (GL_VERTEX_SHADER);
  source              = file_load (vertex_shader);
  glShaderSource (vertexShader, 1, (const char *const *)&source, NULL);
  compile_shader (vertexShader);
  free (source);

  // geometry shader (optionnal)
  GLuint geometryShader;
  if (geometry_shader != NULL) {
    geometryShader = glCreateShader (GL_GEOMETRY_SHADER);
    source         = file_load (geometry_shader);
    glShaderSource (geometryShader, 1, (const char *const *)&source, NULL);
    compile_shader (geometryShader);
    free (source);
  }

  // fragment shader
  GLuint fragmentShader = glCreateShader (GL_FRAGMENT_SHADER);
  source                = file_load (fragment_shader);
  glShaderSource (fragmentShader, 1, (const char *const *)&source, NULL);
  compile_shader (fragmentShader);
  free (source);

  // link shaders
  GLuint program = glCreateProgram ();
  glAttachShader (program, vertexShader);
  if (geometry_shader != NULL)
    glAttachShader (program, geometryShader);
  glAttachShader (program, fragmentShader);
  link_program (program);

  glDeleteShader (vertexShader);
  glDeleteShader (fragmentShader);

  return program;
}

void mesh3d_shader_get_uniform_loc (GLuint program, const char *name,
                                    GLuint *location)
{
  GLuint loc = glGetUniformLocation (program, name);
  check (loc != GL_INVALID_INDEX, "glGetUniformLocation for %s", name);
  *location = loc;
}

void mesh3d_shader_bind_uniform_buf (GLuint program, const char *name,
                                     GLuint blockbinding)
{
  GLuint index = glGetUniformBlockIndex (program, name);
  check (index != GL_INVALID_INDEX, "glGetUniformBlockIndex for %s", name);

  // Link Binding indices to binding points
  glUniformBlockBinding (program, index, blockbinding);
}
