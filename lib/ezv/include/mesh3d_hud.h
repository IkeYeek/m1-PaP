#ifndef MESH3D_HUD_H
#define MESH3D_HUD_H

#include "mesh3d_ctx.h"

#define MAX_HUDS 8
#define MAX_DIGITS 16

typedef struct
{
  int display[MAX_DIGITS];
  int valid;
  int active;
} hud_t;


#endif