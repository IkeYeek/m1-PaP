#ifndef EZV_SDL_GL_H
#define EZV_SDL_GL_H

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
