#ifndef EZV_HUD_H
#define EZV_HUD_H

struct ezv_ctx_s;
typedef struct ezv_ctx_s *ezv_ctx_t;

void ezv_hud_init (ezv_ctx_t ctx);
void ezv_hud_display (ezv_ctx_t ctx);

int ezv_hud_alloc (ezv_ctx_t ctx);
void ezv_hud_free (ezv_ctx_t ctx, int hud);
void ezv_hud_toggle (ezv_ctx_t ctx, int hud);
void ezv_hud_on (ezv_ctx_t ctx, int hud);
void ezv_hud_off (ezv_ctx_t ctx, int hud);
void ezv_hud_set (ezv_ctx_t ctx, int hud, char *format, ...);

#endif
