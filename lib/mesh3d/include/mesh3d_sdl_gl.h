#ifndef MESH3D_SDL_GL_H_IS_DEF
#define MESH3D_SDL_GL_H_IS_DEF

#ifndef USE_GLAD
#include <OpenGL/gl3.h>
// gl3.h must be included before SDL.h
#include <SDL2/SDL.h>

#else

#include <glad/glad.h>
// glad.h should be included before SDL.h
#include <SDL2/SDL.h>

#endif

#endif
