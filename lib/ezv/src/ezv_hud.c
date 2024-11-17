#include <cglm/cglm.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "error.h"
#include "ezv_textures.h"
#include "ezv_ctx.h"
#include "ezv_hud.h"
#include "ezv_sdl_gl.h"
#include "ezv_shader.h"
#include "stb_image.h"

#define MAX_HUDS 8
#define MAX_DIGITS 32

static const float DIGIT_W = 10.0f;
static const float DIGIT_H = 20.0f;

typedef struct
{
  int display[MAX_DIGITS];
  unsigned len;
  int valid;
  int active;
} hud_t;

typedef struct hud_ctx_s
{
  hud_t hud[MAX_HUDS];
  GLuint UBO_HUD, VBO_HUD, VAO_HUD, EBO_HUD; // For on-screen head-up display
  GLuint VBO_BACKG, VAO_BACKG;               // For background
  GLuint hud_shader, backg_shader;
  GLuint hud_digitex_loc, hud_digits_loc, hud_line_loc;
  GLuint digitTexture;
} hud_ctx_t;

static struct
{
  float digit_width;
  float x_spacing;
  float y_spacing;
} HudInfo;

static unsigned char *texture_data = NULL;
static int texture_width      = -1;
static int texture_height     = -1;

static void load_texture_once (void)
{
  int nrChannels;
  char file[1024];

  if (texture_data != NULL)
    return; // texture already loaded

#ifdef WHITE_BACKGROUND
  sprintf (file, "%s/img/ascii-black.png", ezv_prefix ? ezv_prefix : ".");
#else
  sprintf (file, "%s/img/ascii.png", ezv_prefix ? ezv_prefix : ".");
#endif
  stbi_set_flip_vertically_on_load (true);
  texture_data = stbi_load (file, &texture_width, &texture_height, &nrChannels, 0);
  if (texture_data == NULL)
    exit_with_error ("Cannot open %s", file);
  stbi_set_flip_vertically_on_load (false);

  HudInfo.digit_width = DIGIT_W / (float)texture_width;
  HudInfo.x_spacing   = DIGIT_W + 2.0f;
  HudInfo.y_spacing   = DIGIT_H + 4.0f;
}

static void renderer_hud_init (ezv_ctx_t ctx)
{
  ezv_switch_to_context (ctx);

  hud_ctx_t *hudctx = ctx->hud_ctx;

  load_texture_once ();

  glGenBuffers (1, &hudctx->UBO_HUD);
  glBindBuffer (GL_UNIFORM_BUFFER, hudctx->UBO_HUD);
  glBufferData (GL_UNIFORM_BUFFER, sizeof (HudInfo), &HudInfo, GL_STATIC_DRAW);
  glBindBufferBase (GL_UNIFORM_BUFFER, BINDING_POINT_HUDINFO, hudctx->UBO_HUD);

  glGenTextures (1, &hudctx->digitTexture);
  glActiveTexture (GL_TEXTURE0 + EVZ_HUD_TEXTURE_NUM);
  glBindTexture (GL_TEXTURE_2D, hudctx->digitTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, texture_width, texture_height, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, texture_data);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  float vertices[] = {
      // 2D positions     // tex coord
      0.0f,    0.0f,    0.0f,
      1.0f, // top left
      0.0f,    DIGIT_H, 0.0f,
      0.0f, // bottom left
      DIGIT_W, 0.0f,    HudInfo.digit_width,
      1.0f, // top right
      DIGIT_W, DIGIT_H, HudInfo.digit_width,
      0.0f, // bottom right
  };

  // Warning: use clockwise orientation
  unsigned int indices[] = {
      0, 3, 1, // first triangle
      0, 2, 3  // second triangle
  };

  // bind uniform buffer object to texture EVZ_HUD_TEXTURE_NUM
  glProgramUniform1i (hudctx->hud_shader, hudctx->hud_digitex_loc, EVZ_HUD_TEXTURE_NUM);

  glGenVertexArrays (1, &hudctx->VAO_HUD);
  glGenBuffers (1, &hudctx->VBO_HUD);
  glGenBuffers (1, &hudctx->EBO_HUD);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s)
  glBindVertexArray (hudctx->VAO_HUD);

  glBindBuffer (GL_ARRAY_BUFFER, hudctx->VBO_HUD);
  glBufferData (GL_ARRAY_BUFFER, sizeof (vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, hudctx->EBO_HUD);
  glBufferData (GL_ELEMENT_ARRAY_BUFFER, sizeof (indices), indices,
                GL_STATIC_DRAW);

  // configure vertex attributes(s).
  glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof (float),
                         (void *)0);
  glEnableVertexAttribArray (0);

  // configure texture coordinates
  glVertexAttribPointer (1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof (float),
                         (void *)(2 * sizeof (float)));
  glEnableVertexAttribArray (1);

  // Background
  float backv[] = {
      // 2D positions
      0.0f,   0.0f,   // top left
      0.0f,   100.0f, // bottom left
      100.0f, 0.0f,   // top right
      100.0f, 100.0f, // bottom right
  };

  glGenVertexArrays (1, &hudctx->VAO_BACKG);
  glGenBuffers (1, &hudctx->VBO_BACKG);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s)
  glBindVertexArray (hudctx->VAO_BACKG);

  glBindBuffer (GL_ARRAY_BUFFER, hudctx->VBO_BACKG);
  glBufferData (GL_ARRAY_BUFFER, sizeof (backv), backv, GL_DYNAMIC_DRAW);

  glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, hudctx->EBO_HUD);

  // configure vertex attributes(s).
  glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof (float),
                         (void *)0);
  glEnableVertexAttribArray (0);
}

void ezv_hud_display (ezv_ctx_t ctx)
{
  ezv_switch_to_context (ctx);

  int y  = 0;
  int dt = glIsEnabled (GL_DEPTH_TEST);

  hud_ctx_t *hudctx = ctx->hud_ctx;

  if (dt)
    glDisable (GL_DEPTH_TEST);

  for (int h = 0; h < MAX_HUDS; h++) {
    float backv[8];

    if (!hudctx->hud[h].valid || !hudctx->hud[h].active || !hudctx->hud[h].len)
      continue;

    for (int v = 0; v < 4; v++) {
      backv[2 * v] =
          (v < 2) ? 5.0f : 5.0f + hudctx->hud[h].len * HudInfo.x_spacing;
      backv[2 * v + 1] =
          5.0f + y * HudInfo.y_spacing + ((v & 1) ? DIGIT_H : 0.0f);
    }

    glBindBuffer (GL_ARRAY_BUFFER, hudctx->VBO_BACKG);
    glBufferSubData (GL_ARRAY_BUFFER, 0, 8 * sizeof (float), backv);

    glUseProgram (hudctx->backg_shader);

    glBindVertexArray (hudctx->VAO_BACKG);
    glDrawElements (GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glUseProgram (hudctx->hud_shader);

    // refresh 'digits' uniform data
    glProgramUniform1iv (hudctx->hud_shader, hudctx->hud_digits_loc,
                         hudctx->hud[h].len, hudctx->hud[h].display);

    // tell which line to use
    glProgramUniform1i (hudctx->hud_shader, hudctx->hud_line_loc, y++);

    glBindVertexArray (hudctx->VAO_HUD);
    glDrawElementsInstanced (GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0,
                             hudctx->hud[h].len);
  }

  if (dt)
    glEnable (GL_DEPTH_TEST);
}

void ezv_hud_init (ezv_ctx_t ctx)
{
  hud_ctx_t *hudctx = malloc (sizeof (hud_ctx_t));
  ctx->hud_ctx      = hudctx;

  hudctx->UBO_HUD      = 0;
  hudctx->VBO_HUD      = 0;
  hudctx->VAO_HUD      = 0;
  hudctx->EBO_HUD      = 0;
  hudctx->digitTexture = 0;
  hudctx->VAO_BACKG    = 0;
  hudctx->VBO_BACKG    = 0;

  hudctx->hud_shader = ezv_shader_create ("hud/hud.vs", NULL, "hud/hud.fs");
  hudctx->backg_shader =
      ezv_shader_create ("hud/backg.vs", NULL, "hud/backg.fs");

  ezv_shader_get_uniform_loc (hudctx->hud_shader, "digitTexture",
                              &hudctx->hud_digitex_loc);
  ezv_shader_get_uniform_loc (hudctx->hud_shader, "digits",
                              &hudctx->hud_digits_loc);
  ezv_shader_get_uniform_loc (hudctx->hud_shader, "line",
                              &hudctx->hud_line_loc);

  ezv_shader_bind_uniform_buf (hudctx->hud_shader, "HudInfo",
                               BINDING_POINT_HUDINFO);

  ezv_shader_bind_uniform_buf (hudctx->hud_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (hudctx->backg_shader, "Matrices",
                               BINDING_POINT_MATRICES);

  for (int h = 0; h < MAX_HUDS; h++)
    hudctx->hud[h].valid = 0;

  renderer_hud_init (ctx);
}

int ezv_hud_alloc (ezv_ctx_t ctx)
{
  hud_ctx_t *hudctx = ctx->hud_ctx;

  for (int h = 0; h < MAX_HUDS; h++)
    if (!hudctx->hud[h].valid) {
      hudctx->hud[h].valid  = 1;
      hudctx->hud[h].active = 0;
      hudctx->hud[h].len    = 0;
      return h;
    }

  return -1;
}

static void check_validity (int hud, hud_ctx_t *hudctx)
{
  if (hud < 0 || hud >= MAX_HUDS || !hudctx->hud[hud].valid)
    exit_with_error ("Hud %d is invalid", hud);
}

void ezv_hud_free (ezv_ctx_t ctx, int hud)
{
  hud_ctx_t *hudctx = ctx->hud_ctx;

  check_validity (hud, hudctx);

  hudctx->hud[hud].valid = 0;
}

void ezv_hud_toggle (ezv_ctx_t ctx, int hud)
{
  hud_ctx_t *hudctx = ctx->hud_ctx;

  check_validity (hud, hudctx);

  hudctx->hud[hud].active ^= 1;
}

void ezv_hud_on (ezv_ctx_t ctx, int hud)
{
  hud_ctx_t *hudctx = ctx->hud_ctx;

  check_validity (hud, hudctx);

  hudctx->hud[hud].active = 1;
}

void ezv_hud_off (ezv_ctx_t ctx, int hud)
{
  hud_ctx_t *hudctx = ctx->hud_ctx;

  check_validity (hud, hudctx);

  hudctx->hud[hud].active = 0;
}

void ezv_hud_set (ezv_ctx_t ctx, int hud, char *format, ...)
{
  char buffer[MAX_DIGITS + 1];
  int i = 0;

  hud_ctx_t *hudctx = ctx->hud_ctx;

  check_validity (hud, hudctx);

  if (format != NULL) {
    va_list argptr;
    va_start (argptr, format);
    vsnprintf (buffer, MAX_DIGITS + 1, format, argptr);
    va_end (argptr);

    for (; buffer[i] != 0; i++)
      hudctx->hud[hud].display[i] = buffer[i] - ' ';
  }

  hudctx->hud[hud].len = i;
}
