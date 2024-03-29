#include <stdarg.h>

#include "error.h"
#include "mesh3d_ctx.h"


void mesh3d_hud_init (mesh3d_ctx_t ctx)
{
  for (int h = 0; h < MAX_HUDS; h++)
    ctx->hud[h].valid = 0;
}

int mesh3d_hud_alloc (mesh3d_ctx_t ctx)
{
  for (int h = 0; h < MAX_HUDS; h++)
    if (!ctx->hud[h].valid) {
      ctx->hud[h].valid  = 1;
      ctx->hud[h].active = 0;
      bzero (&ctx->hud[h].display, sizeof (ctx->hud[h].display));
      return h;
    }

  return -1;
}

void mesh3d_hud_free (mesh3d_ctx_t ctx, int hud)
{
  if (hud < 0 || hud >= MAX_HUDS || !ctx->hud[hud].valid)
    exit_with_error ("Hud %d is invalid", hud);

  ctx->hud[hud].valid = 0;
}

void mesh3d_hud_toggle (mesh3d_ctx_t ctx, int hud)
{
  if (hud < 0 || hud >= MAX_HUDS || !ctx->hud[hud].valid)
    exit_with_error ("Hud %d is invalid", hud);

  ctx->hud[hud].active ^= 1;
}

void mesh3d_hud_on (mesh3d_ctx_t ctx, int hud)
{
  if (hud < 0 || hud >= MAX_HUDS || !ctx->hud[hud].valid)
    exit_with_error ("Hud %d is invalid", hud);

  ctx->hud[hud].active = 1;
}

void mesh3d_hud_off (mesh3d_ctx_t ctx, int hud)
{
  if (hud < 0 || hud >= MAX_HUDS || !ctx->hud[hud].valid)
    exit_with_error ("Hud %d is invalid", hud);

  ctx->hud[hud].active = 0;
}

void mesh3d_hud_set (mesh3d_ctx_t ctx, int hud, char *format, ...)
{
  int i = 0;
  char buffer[MAX_DIGITS + 1];

  if (hud < 0 || hud >= MAX_HUDS || !ctx->hud[hud].valid)
    exit_with_error ("Hud %d is invalid", hud);

  if (format != NULL) {
    va_list argptr;
    va_start (argptr, format);
    vsnprintf (buffer, MAX_DIGITS + 1, format, argptr);
    va_end (argptr);

    // FIXME: should be ctx-hud[h].display
    for (; buffer[i] != 0; i++)
      ctx->hud[hud].display[i] = buffer[i] - ' ';
  }

  // add spaces
  for (; i < MAX_DIGITS; i++)
    ctx->hud[hud].display[i] = 0;
}