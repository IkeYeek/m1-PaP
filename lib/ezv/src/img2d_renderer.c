#include <cglm/cglm.h>
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

#include "ezv_img2d_object.h"
#include "img2d_renderer.h"

// #define WHITE_BACKGROUND 1

static const float DEFAULT_DATA_BRIGHTNESS = 0.8f;

static const float NEARZ = -2.0f;
static const float FARZ  = 2.0f;

typedef struct img2d_render_ctx_s
{
  GLuint UBO_DATACOL;
  GLuint UBO_MAT, UBO_CUST;
  GLuint VBO;       // Vertex Buffer Object (contains vertices)
  GLuint VAO;       // Vertex Array Object
  GLuint VBO_IND;   // Vertex Buffer Object containing triangles (i.e. 3-tuples
                    // indexing vertices)
  GLuint FBO;
  GLuint pickingTexture, dataTexture, cpuTexture;
  GLuint cpu_shader, picking_shader, data_shader, dapu_shader;
  GLuint cpu_imgtex_loc;
  GLuint data_imgtex_loc;
  GLuint dapu_cputex_loc, dapu_datatex_loc, dapu_brightness_loc;
} img2d_render_ctx_t;

static float INITIAL_SCALE    = 1.0f;
static float INITIAL_OFFSET_Y = 0.0f;
static float INITIAL_OFFSET_X = 0.0f;

static GLfloat offset_x = 0.0f;
static GLfloat offset_y = 0.0f;
static GLfloat scale_xy = 1.0f;

// Model Matrix
static struct
{
  mat4 mvp;
  mat4 ortho;
  mat4 vp_unclipped;
  mat4 mvp_unclipped;
} Matrices;

static struct
{
  vec4 img_color;
  vec4 cut_color;
} CustomColors;

static mat4 projection, unclipped_proj;

// Normalization operations translate and rescale the object to fit into a
// normalized rectangle centered in (0,0)
static vec3 norm_scale, norm_translate;

void img2d_reset_view (ezv_ctx_t ctx[], unsigned nb_ctx)
{
  offset_x = INITIAL_OFFSET_X;
  offset_y = INITIAL_OFFSET_Y;
  scale_xy = INITIAL_SCALE;

  img2d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, 1.0f);
}

static void img2d_renderer_mvp_init (ezv_ctx_t ctx)
{
  static int done            = 0;
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);
  img2d_obj_t *img           = ezv_img2d_img (ctx);

  // Create matrices and vector once
  if (!done) {
    mat4 model, view;
    GLfloat msize;

    msize = glm_max (img->width, img->height);

    scale_xy = INITIAL_SCALE = (2.0 / msize);
    offset_x = INITIAL_OFFSET_X = -(float)(img->width >> 1);
    offset_y = INITIAL_OFFSET_Y = -(float)(img->height >> 1);
    
    // Compute norm_scale…
    glm_vec3_copy ((vec3){scale_xy, scale_xy, 1.0f}, norm_scale);
    // … and norm_translate, to be used in model matrix
    glm_vec3_copy ((vec3){offset_x, offset_y, 0.0f}, norm_translate);

    glm_mat4_identity (model);
    glm_scale (model, norm_scale);
    glm_translate (model, norm_translate);

    glm_mat4_identity (view);

    float xprop = scale_xy * (img->width >> 1);
    float yprop = scale_xy * (img->height >> 1);

    GLfloat sc_x = (float)ctx->winw / (float)(img->width);
    GLfloat sc_y = (float)ctx->winh / (float)(img->height);

    if (sc_x < sc_y) {
      // x scale is the limiting one
      sc_x = xprop;
      sc_y = (float)(ctx->winh) / (float)(ctx->winw) * xprop;
    } else {
      // y scale is the limiting one
      sc_y = yprop;
      sc_x = (float)(ctx->winw) / (float)(ctx->winh) * yprop;
    }

    glm_ortho (-sc_x, sc_x, sc_y, -sc_y, NEARZ, FARZ, projection);
    glm_mat4_copy (projection, unclipped_proj);

    glm_mat4_mul (projection, view, Matrices.vp_unclipped);
    glm_mat4_mul (Matrices.vp_unclipped, model, Matrices.mvp);
    glm_mat4_copy (Matrices.mvp, Matrices.mvp_unclipped);

    glm_ortho (0.0f, (float)ctx->winw, (float)ctx->winh, 0.0f, -2.0f, 2.0f,
               Matrices.ortho);

    glm_vec4_copy ((vec4){0.5, 0.5, 0.5, 1.0}, CustomColors.img_color);
    glm_vec4_copy ((vec4){1.0, 1.0, 1.0, 1.0}, CustomColors.cut_color);

    done = 1;
  }

  glGenBuffers (1, &renctx->UBO_MAT);
  glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_MAT);
  glBufferData (GL_UNIFORM_BUFFER, sizeof (Matrices), &Matrices,
                GL_DYNAMIC_DRAW);
  glBindBufferBase (GL_UNIFORM_BUFFER, BINDING_POINT_MATRICES, renctx->UBO_MAT);

  glGenBuffers (1, &renctx->UBO_CUST);
  glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_CUST);
  glBufferData (GL_UNIFORM_BUFFER, sizeof (CustomColors), &CustomColors,
                GL_STATIC_DRAW); // Will not change often
  glBindBufferBase (GL_UNIFORM_BUFFER, BINDING_POINT_CUSTOM_COLORS,
                    renctx->UBO_CUST);
}

void img2d_renderer_mvp_update (ezv_ctx_t ctx[], unsigned nb_ctx, float dx,
                                float dy, float dz)
{
  mat4 model, view, tmp;

  offset_x += dx;
  offset_y += dy;
  scale_xy *= dz;

  glm_vec3_copy ((vec3){scale_xy, scale_xy, 1.0f}, norm_scale);
  // … and norm_translate, to be used in model matrix
  glm_vec3_copy ((vec3){offset_x, offset_y, 0.0f}, norm_translate);

  glm_mat4_identity (model);
  glm_scale (model, norm_scale);
  glm_translate (model, norm_translate);

  glm_mat4_identity (view);

  glm_mat4_mul (projection, view, tmp);
  glm_mat4_mul (tmp, model, Matrices.mvp);

  glm_mat4_mul (unclipped_proj, view, Matrices.vp_unclipped);
  glm_mat4_mul (Matrices.vp_unclipped, model, Matrices.mvp_unclipped);

  for (int c = 0; c < nb_ctx; c++) {
    img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx[c]);

    ezv_switch_to_context (ctx[c]);

    glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_MAT);
    glBufferSubData (GL_UNIFORM_BUFFER, 0, sizeof (Matrices), &Matrices);
  }
}

static void img2d_picking_init (ezv_ctx_t ctx)
{
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);

  // Create the FBO for triangle picking
  glGenFramebuffers (1, &renctx->FBO);
  glBindFramebuffer (GL_FRAMEBUFFER, renctx->FBO);

  // Create the texture object for the primitive information buffer
  glGenTextures (1, &renctx->pickingTexture);
  glBindTexture (GL_TEXTURE_2D, renctx->pickingTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RG32F, ctx->winw, ctx->winh, 0, GL_RG,
                GL_FLOAT, NULL);
  glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          renctx->pickingTexture, 0);

  glBindTexture (GL_TEXTURE_2D, 0);
  glBindFramebuffer (GL_FRAMEBUFFER, 0);
}

// called by ctx_create: the img is not defined yet, nor any palette
void img2d_renderer_init (ezv_ctx_t ctx)
{
  ezv_switch_to_context (ctx);

  // configure global opengl state
  // -----------------------------
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable (GL_BLEND);

  // Allocate render_ctx
  img2d_render_ctx_t *renctx = malloc (sizeof (img2d_render_ctx_t));
  renctx->UBO_DATACOL        = 0;
  renctx->UBO_MAT            = 0;
  renctx->UBO_CUST           = 0;
  renctx->VBO                = 0;
  renctx->VAO                = 0;
  renctx->VBO_IND            = 0;
  renctx->pickingTexture     = 0;
  renctx->dataTexture        = 0;
  renctx->cpuTexture         = 0;

  ezv_img2d_set_renderer (ctx, renctx);

  // compile shaders and build program
  renctx->cpu_shader =
      ezv_shader_create ("img2d/generic.vs", NULL, "img2d/generic.fs");
  renctx->data_shader =
      ezv_shader_create ("img2d/generic.vs", NULL, "img2d/generic.fs");
  renctx->dapu_shader =
      ezv_shader_create ("img2d/generic.vs", NULL, "img2d/data_rgba_cpu.fs");
  renctx->picking_shader =
      ezv_shader_create ("img2d/picking.vs", NULL, "img2d/picking.fs");

  // Uniform parameters

  ezv_shader_get_uniform_loc (renctx->data_shader, "dataTexture",
                              &renctx->data_imgtex_loc);

  ezv_shader_get_uniform_loc (renctx->cpu_shader, "dataTexture",
                              &renctx->cpu_imgtex_loc);

  ezv_shader_get_uniform_loc (renctx->dapu_shader, "cpuTexture",
                              &renctx->dapu_cputex_loc);
  ezv_shader_get_uniform_loc (renctx->dapu_shader, "dataTexture",
                              &renctx->dapu_datatex_loc);
  ezv_shader_get_uniform_loc (renctx->dapu_shader, "dataBrightness",
                              &renctx->dapu_brightness_loc);

  // Bind Matrices to all shaders
  ezv_shader_bind_uniform_buf (renctx->cpu_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->data_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->dapu_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->picking_shader, "Matrices",
                               BINDING_POINT_MATRICES);

  if (ctx->picking_enabled)
    img2d_picking_init (ctx);
}

void img2d_renderer_set_img (ezv_ctx_t ctx)
{
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);
  img2d_obj_t *img           = ezv_img2d_img (ctx);

  ezv_switch_to_context (ctx);

  // Initialize 'Matrices'
  img2d_renderer_mvp_init (ctx);

  float vertices[] = {
      // 2D positions     // tex coord
      0.0f,
      0.0f,
      0.0f,
      0.0f, // top left
      0.0f,
      (float)(img->height),
      0.0f,
      1.0f, // bottom left
      (float)(img->width),
      0.0f,
      1.0f,
      0.0f, // top right
      (float)(img->width),
      (float)(img->height),
      1.0f,
      1.0f, // bottom right
  };

  // Warning: use clockwise orientation
  unsigned int indices[] = {
      0, 3, 1, // first triangle
      0, 2, 3  // second triangle
  };

  // configure vertex attributes and misc buffers
  glGenVertexArrays (1, &renctx->VAO);
  glGenBuffers (1, &renctx->VBO);
  glGenBuffers (1, &renctx->VBO_IND);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s)
  glBindVertexArray (renctx->VAO);

  glBindBuffer (GL_ARRAY_BUFFER, renctx->VBO);
  glBufferData (GL_ARRAY_BUFFER, sizeof (vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, renctx->VBO_IND);
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
}

void img2d_renderer_use_cpu_palette (ezv_ctx_t ctx)
{
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);
  img2d_obj_t *img           = ezv_img2d_img (ctx);

  if (!ezv_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not defined");

  ezv_switch_to_context (ctx);

  // Allocate a new 2D texture
  glGenTextures (1, &renctx->cpuTexture);
  glActiveTexture (GL_TEXTURE0 + EZV_CPU_TEXTURE_NUM);

  glBindTexture (GL_TEXTURE_2D, renctx->cpuTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, img->width, img->height, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // bind uniform buffer object to texture #1
  glProgramUniform1i (renctx->cpu_shader, renctx->cpu_imgtex_loc, EZV_CPU_TEXTURE_NUM);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_cputex_loc, EZV_CPU_TEXTURE_NUM);
}

void img2d_renderer_use_data_palette (ezv_ctx_t ctx)
{
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);
  img2d_obj_t *img           = ezv_img2d_img (ctx);

  if (!ezv_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  if (ctx->data_palette.name != EZV_PALETTE_RGBA_PASSTHROUGH)
    exit_with_error ("Only RGBA colors are supported yet");

  ezv_switch_to_context (ctx);

  // Data brightness
  glProgramUniform1f (renctx->dapu_shader, renctx->dapu_brightness_loc,
                      DEFAULT_DATA_BRIGHTNESS);

  // Allocate a new 2D texture
  glGenTextures (1, &renctx->dataTexture);
  glActiveTexture (GL_TEXTURE0 + EZV_DATA_TEXTURE_NUM);

  glBindTexture (GL_TEXTURE_2D, renctx->dataTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, img->width, img->height, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // bind uniform buffer object to texture #3
  glProgramUniform1i (renctx->data_shader, renctx->data_imgtex_loc, EZV_DATA_TEXTURE_NUM);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_datatex_loc, EZV_DATA_TEXTURE_NUM);
}

void img2d_set_data_brightness (ezv_ctx_t ctx, float brightness)
{
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);

  ezv_switch_to_context (ctx);

  glProgramUniform1f (renctx->dapu_shader, renctx->dapu_brightness_loc,
                      brightness);
}

void img2d_get_shareable_buffer_ids (ezv_ctx_t ctx, int buffer_ids[])
{
  if (!ezv_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);

  ezv_switch_to_context (ctx);

  buffer_ids[0] = renctx->dataTexture;
}

void img2d_set_data_colors (ezv_ctx_t ctx, void *values)
{
  if (!ezv_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  img2d_obj_t *img           = ezv_img2d_img (ctx);

  ezv_switch_to_context (ctx);

  glActiveTexture (GL_TEXTURE0 + EZV_DATA_TEXTURE_NUM);
  glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, img->width, img->height, GL_RGBA,
                   GL_UNSIGNED_BYTE, values);
}

void img2d_renderer_do_picking (ezv_ctx_t ctx, int mousex, int mousey, int *x,
                                int *y)
{
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);

  ezv_switch_to_context (ctx);

  glUseProgram (renctx->picking_shader);

  glBindFramebuffer (GL_DRAW_FRAMEBUFFER, renctx->FBO); // ON

  glDisable (GL_BLEND);
  glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
  glClear (GL_COLOR_BUFFER_BIT);

  // draw triangles
  glDrawElements (GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

  glBindFramebuffer (GL_DRAW_FRAMEBUFFER, 0); // OFF

  // Read texture at position (mouse_x, mouse_y)
  glBindFramebuffer (GL_READ_FRAMEBUFFER, renctx->FBO);
  glReadBuffer (GL_COLOR_ATTACHMENT0);

  float pixel[2] = {0.0f, 0.0f};

  glReadPixels (mousex, ctx->winh - mousey - 1, 1, 1, GL_RG, GL_FLOAT, pixel);

  glReadBuffer (GL_NONE);
  glBindFramebuffer (GL_READ_FRAMEBUFFER, 0);

  glEnable (GL_BLEND);

  *x = (int)(pixel[0] - 1.0f);
  *y = (int)(pixel[1] - 1.0f);
}

void img2d_render (ezv_ctx_t ctx)
{
  img2d_obj_t *img           = ezv_img2d_img (ctx);
  img2d_render_ctx_t *renctx = ezv_img2d_renderer (ctx);

  ezv_switch_to_context (ctx);

#ifdef WHITE_BACKGROUND
  glClearColor (1.0f, 1.f, 1.0f, 1.0f);
#else
  glClearColor (0.0f, 0.2f, 0.2f, 1.0f);
#endif
  glClear (GL_COLOR_BUFFER_BIT |
           GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

  if (ezv_palette_is_defined (&ctx->cpu_palette)) {

    glActiveTexture (GL_TEXTURE0 + EZV_CPU_TEXTURE_NUM);
    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, img->width, img->height, GL_RGBA,
                   GL_UNSIGNED_BYTE, ctx->cpu_colors);

    if (ezv_palette_is_defined (&ctx->data_palette))
      // Both CPU and Data colors are used
      glUseProgram (renctx->dapu_shader);
    else
      // CPU colors only
      glUseProgram (renctx->cpu_shader);

    glDrawElements (GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

  } else if (ezv_palette_is_defined (&ctx->data_palette)) {
    // Data colors only
    glUseProgram (renctx->data_shader);

    glDrawElements (GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  }

  if (ctx->hud_enabled) {
    // On-Screen Head Up Display
    ezv_hud_display (ctx);
    glBindVertexArray (renctx->VAO);
  }

  SDL_GL_SwapWindow (ctx->win);
}

int img2d_renderer_zoom (ezv_ctx_t ctx[], unsigned nb_ctx, unsigned shift_mod,
                         unsigned in)
{
  // Zoom in
  img2d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f,
                             in ? +1.02f : 1.0f / 1.02f);
  return 1;
}

int img2d_renderer_motion (ezv_ctx_t ctx[], unsigned nb_ctx, int dx, int dy,
                           unsigned wheel)
{
  float deltax = ((float)dx / (float)ctx[0]->winw) * ((2.0 / scale_xy));
  float deltay = ((float)dy / (float)ctx[0]->winh) * ((2.0 / scale_xy));

  if (wheel)
    img2d_renderer_mvp_update (ctx, nb_ctx, deltax * 10.0f, deltay * 10.0f,
                               1.0f);
  else {
    img2d_renderer_mvp_update (ctx, nb_ctx, deltax, deltay, 1.0f);
  }

  return 1;
}
