#include <cglm/cglm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "error.h"
#include "mesh3d_ctx.h"
#include "mesh3d_renderer.h"
#include "mesh3d_sdl_gl.h"
#include "mesh3d_shader.h"
#include "stb_image.h"

enum
{
  BINDING_POINT_COLORBUF,
  BINDING_POINT_MATRICES,
  BINDING_POINT_CUSTOM_COLORS,
  BINDING_POINT_CLIPPING,
  BINDING_POINT_HUDINFO
};

static const float DEFAULT_DATA_BRIGHTNESS = 0.8f;
static const float DIGIT_W = 10.0f;
static const float DIGIT_H = 20.0f;

const float NEARZ = 0.2f;
const float FARZ  = 5.0f;

typedef struct render_ctx_s
{
  GLuint UBO_DATACOL;
  GLuint UBO_MAT, UBO_CUST, UBO_CLIP, UBO_HUD;
  GLuint VBO;                       // Vertex Buffer Object (contains vertices)
  GLuint VAO;                       // Vertex Array Object
  GLuint VBO_HUD, VAO_HUD, EBO_HUD; // For on-screen head-up display
  GLuint VBO_ZPLANE, VAO_ZPLANE, EBO_ZPLANE; // for zplane clipping
  GLuint VBO_IND;  // Vertex Buffer Object containing triangles (i.e. 3-tuples
                   // indexing vertices)
  GLuint TBO_INFO; // Texture Buffer Object containing per-triangle info
                   // (cellno, drop, edgen)
  GLuint TBO_COL;  // Texture Buffer Object containing per-cell indexes to cpu
                   // color palette
  GLuint TBO_DATACOL[2];
  unsigned current_datacol;
  GLuint tex_tinfo;   // GL Texture associated to TBO_INFO
  GLuint tex_color;   // TBO_COL
  GLuint tex_datacol; // TBO_DATACOL
  GLuint FBO;
  GLuint pickingTexture, depthTexture, digitTexture;
  GLuint cpu_shader, picking_shader, data_shader, hud_shader, dapu_shader,
      clipping_shader, cut_shader;
  GLuint cpu_colors_loc, cpu_info_loc;
  GLuint data_values_loc, data_info_loc, data_palettesize_loc;
  GLuint hud_digitex_loc, hud_digits_loc, hud_line_loc;
  GLuint dapu_colors_loc, dapu_info_loc, dapu_values_loc, dapu_palettesize_loc, dapu_brightness_loc;
  GLuint cut_info_loc;
} mesh3d_render_ctx_t;

const float INITIAL_TRANSLATE_VAL = -1.5f;
const float INITIAL_ROTATE_Y      = -90.0f;
const float INITIAL_ROTATE_X      = +15.0f;

static GLfloat rotate_x    = INITIAL_ROTATE_X;
static GLfloat rotate_y    = INITIAL_ROTATE_Y;
static GLfloat translate_z = INITIAL_TRANSLATE_VAL;

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
  vec4 mesh_color;
  vec4 cut_color;
} CustomColors;

static struct
{
  float clipping_zcoord;
  float clipping_zproj;
  int clipping_active;
  float width;
  float height;
} Clipping;

static struct
{
  float digit_width;
  float x_spacing;
  float y_spacing;
} HudInfo;

static mat4 projection, unclipped_proj;

// Normalization operations translate and rescale the object to fit into a
// normalized bounding-box centered in (0,0,0)
static vec3 norm_scale, norm_translate;

void mesh3d_switch_to_context (mesh3d_ctx_t ctx)
{
  SDL_GL_MakeCurrent (ctx->win, ctx->glcontext);
}

static void set_transformed_zplane (void)
{
  // We must pass the projected z value of the zplane…
  vec4 center = {0.0, 0.0, Clipping.clipping_zcoord, 1.0};
  vec4 proj;

  glm_mat4_mulv (Matrices.vp_unclipped, center, proj);

  Clipping.clipping_zproj = proj[3];
}

void mesh3d_reset_view (mesh3d_ctx_t ctx[], unsigned nb_ctx)
{
  rotate_x    = INITIAL_ROTATE_X;
  rotate_y    = INITIAL_ROTATE_Y;
  translate_z = INITIAL_TRANSLATE_VAL;

  Clipping.clipping_zcoord = 0.0f;

  mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, 0.0f);
}

static void mesh3d_renderer_mvp_init (mesh3d_ctx_t ctx)
{
  static int done             = 0;
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  // Create matrices and vector once
  if (!done) {
    bbox_t *bbox = &ctx->mesh->bbox;
    GLfloat msize;
    vec3 dim;
    mat4 model, view;

    glm_vec3_sub (bbox->max, bbox->min, dim);
    msize = glm_max (glm_max (dim[0], dim[1]), dim[2]);

    // Compute norm_scale…
    vec3 scale = {msize, msize, msize};
    glm_vec3_div ((vec3){1.0, 1.0, 1.0}, scale, norm_scale);
    // … and norm_translate, to be used in model matrix
    glm_vec3_add (bbox->min, bbox->max, norm_translate);
    glm_vec3_div (norm_translate, (vec3){2.0, 2.0, 2.0}, norm_translate);
    glm_vec3_negate (norm_translate);

    glm_mat4_identity (model);
    glm_rotate (model, glm_rad (rotate_x), (vec3){1.0f, 0.0f, 0.0f});
    glm_rotate (model, glm_rad (rotate_y), (vec3){0.0f, 1.0f, 0.0f});
    glm_scale (model, norm_scale);
    glm_translate (model, norm_translate);

    glm_mat4_identity (view);
    glm_translate (view, (vec3){0.0, 0.0, translate_z});

    glm_perspective (glm_rad (45.0), (float)ctx->winw / (float)ctx->winh, NEARZ,
                     FARZ, projection);
    glm_mat4_copy (projection, unclipped_proj);

    glm_mat4_mul (projection, view, Matrices.vp_unclipped);
    glm_mat4_mul (Matrices.vp_unclipped, model, Matrices.mvp);
    glm_mat4_copy (Matrices.mvp, Matrices.mvp_unclipped);

    glm_ortho (0.0f, (float)ctx->winw, (float)ctx->winh, 0.0f, NEARZ, FARZ,
               Matrices.ortho);

    if (ctx->mesh->mesh_type == MESH3D_TYPE_SURFACE)
      // lighgrey mesh for surfaces
      glm_vec4_copy ((vec4){0.5, 0.5, 0.5, 1.0}, CustomColors.mesh_color);
    else
      // black mesh for volumes
      glm_vec4_copy ((vec4){0.3, 0.3, 0.3, 1.0}, CustomColors.mesh_color);
    glm_vec4_copy ((vec4){1.0, 1.0, 1.0, 1.0}, CustomColors.cut_color);

    Clipping.clipping_zcoord = 0.0;
    Clipping.clipping_active = 0;
    Clipping.width           = ctx->winw / 2;
    Clipping.height          = ctx->winh / 2;

    set_transformed_zplane ();

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

  glGenBuffers (1, &renctx->UBO_CLIP);
  glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_CLIP);
  glBufferData (GL_UNIFORM_BUFFER, sizeof (Clipping), &Clipping,
                GL_DYNAMIC_DRAW);
  glBindBufferBase (GL_UNIFORM_BUFFER, BINDING_POINT_CLIPPING,
                    renctx->UBO_CLIP);
}

void mesh3d_renderer_mvp_update (mesh3d_ctx_t ctx[], unsigned nb_ctx, float dx,
                                 float dy, float dz)
{
  mat4 model, view, tmp;

  rotate_x += dx;
  rotate_y += dy;
  translate_z += dz;

  glm_mat4_identity (model);
  glm_rotate (model, glm_rad (rotate_x), (vec3){1.0f, 0.0f, 0.0f});
  glm_rotate (model, glm_rad (rotate_y), (vec3){0.0f, 1.0f, 0.0f});
  glm_scale (model, norm_scale);
  glm_translate (model, norm_translate);

  glm_mat4_identity (view);
  glm_translate (view, (vec3){0.0, 0.0, translate_z});

  if (ctx[0]->clipping_enabled)
    // we must re-compute projection to take z clipping plane into account
    glm_perspective (glm_rad (45.0), (float)ctx[0]->winw / (float)ctx[0]->winh,
                     ctx[0]->clipping_active
                         ? -translate_z - Clipping.clipping_zcoord
                         : NEARZ,
                     FARZ, projection);

  glm_mat4_mul (projection, view, tmp);
  glm_mat4_mul (tmp, model, Matrices.mvp);

  glm_mat4_mul (unclipped_proj, view, Matrices.vp_unclipped);
  glm_mat4_mul (Matrices.vp_unclipped, model, Matrices.mvp_unclipped);

  set_transformed_zplane ();

  for (int c = 0; c < nb_ctx; c++) {
    mesh3d_switch_to_context (ctx[c]);

    glBindBuffer (GL_UNIFORM_BUFFER, ctx[c]->render_ctx->UBO_MAT);
    glBufferSubData (GL_UNIFORM_BUFFER, 0, sizeof (Matrices), &Matrices);

    glBindBuffer (GL_UNIFORM_BUFFER, ctx[c]->render_ctx->UBO_CLIP);
    glBufferSubData (GL_UNIFORM_BUFFER, 0, sizeof (Clipping), &Clipping);
  }
}

void mesh3d_renderer_zplane_update (mesh3d_ctx_t ctx[], unsigned nb_ctx,
                                    float dz)
{
  Clipping.clipping_zcoord += dz;
  Clipping.clipping_active = ctx[0]->clipping_active;

  mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, 0.0f);
}

static void mesh3d_renderer_hud_init (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  int width, height, nrChannels;
  char file[1024];
  unsigned char *data = NULL;

  sprintf (file, "%s/img/ascii.png", mesh3d_prefix ? mesh3d_prefix : ".");
  stbi_set_flip_vertically_on_load (true);
  data = stbi_load (file, &width, &height, &nrChannels, 0);
  if (data == NULL)
    exit_with_error ("Cannot open %s", file);

  HudInfo.digit_width = DIGIT_W / (float)width;
  HudInfo.x_spacing   = DIGIT_W + 2.0f;
  HudInfo.y_spacing   = DIGIT_H + 4.0f;

  glGenBuffers (1, &renctx->UBO_HUD);
  glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_HUD);
  glBufferData (GL_UNIFORM_BUFFER, sizeof (HudInfo), &HudInfo, GL_STATIC_DRAW);
  glBindBufferBase (GL_UNIFORM_BUFFER, BINDING_POINT_HUDINFO, renctx->UBO_HUD);

  glGenTextures (1, &renctx->digitTexture);
  glActiveTexture (GL_TEXTURE4);
  glBindTexture (GL_TEXTURE_2D, renctx->digitTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, data);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  stbi_image_free (data);

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

  // bind uniform buffer object to texture #4
  glProgramUniform1i (renctx->hud_shader, renctx->hud_digitex_loc, 4);

  glGenVertexArrays (1, &renctx->VAO_HUD);
  glGenBuffers (1, &renctx->VBO_HUD);
  glGenBuffers (1, &renctx->EBO_HUD);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s)
  glBindVertexArray (renctx->VAO_HUD);

  glBindBuffer (GL_ARRAY_BUFFER, renctx->VBO_HUD);
  glBufferData (GL_ARRAY_BUFFER, sizeof (vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, renctx->EBO_HUD);
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

  // Not useful
  // bzero (the_digits, sizeof (the_digits));
}

static void mesh3d_hud_display (mesh3d_ctx_t ctx)
{
  int y = 0;

  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  glDisable (GL_DEPTH_TEST);

  glUseProgram (renctx->hud_shader);

  glBindVertexArray (renctx->VAO_HUD);

  for (int h = 0; h < MAX_HUDS; h++) {
    if (!ctx->hud[h].valid || !ctx->hud[h].active)
      continue;

    // refresh 'digits' uniform data
    glProgramUniform1iv (renctx->hud_shader, renctx->hud_digits_loc, MAX_DIGITS,
                         ctx->hud[h].display);

    // tell which line to use
    glProgramUniform1i (renctx->hud_shader, renctx->hud_line_loc, y++);

    glDrawElementsInstanced (GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, MAX_DIGITS);
  }

  glBindVertexArray (renctx->VAO);

  glEnable (GL_DEPTH_TEST);
}

static void mesh3d_zplane_init (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  float vertices[] = {
      // 2D positions
      -.65f, .65f * (float)ctx->winh / (float)ctx->winw, //
      -.65f, -.65f * (float)ctx->winh / (float)ctx->winw, //
      .65f, -.65f * (float)ctx->winh / (float)ctx->winw, //
      .65f, .65f * (float)ctx->winh / (float)ctx->winw,
  };

  glGenVertexArrays (1, &renctx->VAO_ZPLANE);
  glGenBuffers (1, &renctx->VBO_ZPLANE);
  glGenBuffers (1, &renctx->EBO_ZPLANE);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s)
  glBindVertexArray (renctx->VAO_ZPLANE);

  glBindBuffer (GL_ARRAY_BUFFER, renctx->VBO_ZPLANE);
  glBufferData (GL_ARRAY_BUFFER, sizeof (vertices), vertices, GL_STATIC_DRAW);

  // configure vertex attributes(s).
  glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof (float),
                         (void *)0);
  glEnableVertexAttribArray (0);
}

static void mesh3d_zplane_display (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;
  mesh3d_obj_t *mesh          = ctx->mesh;

  glDisable (GL_DEPTH_TEST);

  glUseProgram (renctx->clipping_shader);

  glBindVertexArray (renctx->VAO_ZPLANE);
  glDrawArrays (GL_LINE_LOOP, 0, 4);

  glBindVertexArray (renctx->VAO);

  // Draw cut
  glUseProgram (renctx->cut_shader);

  glDrawElements (GL_TRIANGLES, mesh->nb_triangles * 3, GL_UNSIGNED_INT, 0);

  glEnable (GL_DEPTH_TEST);
}

static void mesh3d_picking_init (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  // Create the FBO for or triangle picking
  glGenFramebuffers (1, &renctx->FBO);
  glBindFramebuffer (GL_FRAMEBUFFER, renctx->FBO);

  // Create the texture object for the primitive information buffer
  glGenTextures (1, &renctx->pickingTexture);
  glBindTexture (GL_TEXTURE_2D, renctx->pickingTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_R32F, ctx->winw, ctx->winh, 0, GL_RED,
                GL_FLOAT, NULL);
  glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                          renctx->pickingTexture, 0);

  // Create the texture object for the depth buffer
  glGenTextures (1, &renctx->depthTexture);
  glBindTexture (GL_TEXTURE_2D, renctx->depthTexture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, ctx->winw, ctx->winh, 0,
                GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  glFramebufferTexture2D (GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                          renctx->depthTexture, 0);

  glBindTexture (GL_TEXTURE_2D, 0);
  glBindFramebuffer (GL_FRAMEBUFFER, 0);
}

// called by ctx_create: the mesh is not defined yet, nor any palette
void mesh3d_renderer_init (mesh3d_ctx_t ctx)
{
  mesh3d_switch_to_context (ctx);

  // configure global opengl state
  // -----------------------------
  glEnable (GL_DEPTH_TEST);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable (GL_BLEND);
  glEnable (GL_LINE_SMOOTH);
  glLineWidth (.75);

  // glEnable(GL_CLIP_DISTANCE0);

  // Allocate render_ctx
  mesh3d_render_ctx_t *renctx = malloc (sizeof (mesh3d_render_ctx_t));
  renctx->UBO_DATACOL         = 0;
  renctx->UBO_MAT             = 0;
  renctx->UBO_CUST            = 0;
  renctx->UBO_CLIP            = 0;
  renctx->UBO_HUD             = 0;
  renctx->VBO                 = 0;
  renctx->VAO                 = 0;
  renctx->VBO_HUD             = 0;
  renctx->VAO_HUD             = 0;
  renctx->EBO_HUD             = 0;
  renctx->VBO_ZPLANE          = 0;
  renctx->VAO_ZPLANE          = 0;
  renctx->EBO_ZPLANE          = 0;
  renctx->VBO_IND             = 0;
  renctx->TBO_INFO            = 0;
  renctx->TBO_COL             = 0;
  renctx->current_datacol     = 0;
  renctx->TBO_DATACOL[0]      = 0;
  renctx->TBO_DATACOL[1]      = 0;
  renctx->tex_tinfo           = 0;
  renctx->tex_color           = 0;
  renctx->depthTexture        = 0;
  renctx->pickingTexture      = 0;
  renctx->digitTexture        = 0;

  ctx->render_ctx = renctx;

  // compile shaders and build program
  renctx->cpu_shader =
      mesh3d_shader_create ("generic.vs", "cpu.gs", "generic.fs");
  renctx->data_shader =
      mesh3d_shader_create ("generic.vs", "data.gs", "generic.fs");
  renctx->dapu_shader =
      mesh3d_shader_create ("generic.vs", "cpu_data.gs", "generic.fs");
  renctx->picking_shader =
      mesh3d_shader_create ("generic.vs", "generic.gs", "picking.fs");
  renctx->hud_shader = mesh3d_shader_create ("hud.vs", NULL, "hud.fs");
  renctx->clipping_shader =
      mesh3d_shader_create ("clipping.vs", NULL, "cut.fs");
  renctx->cut_shader = mesh3d_shader_create ("cut.vs", "cut.gs", "cut.fs");

  // Uniform parameters
  mesh3d_shader_get_uniform_loc (renctx->cpu_shader, "TriangleInfo",
                                 &renctx->cpu_info_loc);
  mesh3d_shader_get_uniform_loc (renctx->cpu_shader, "RGBAColors",
                                 &renctx->cpu_colors_loc);

  mesh3d_shader_get_uniform_loc (renctx->data_shader, "TriangleInfo",
                                 &renctx->data_info_loc);
  mesh3d_shader_get_uniform_loc (renctx->data_shader, "Values",
                                 &renctx->data_values_loc);
  mesh3d_shader_get_uniform_loc (renctx->data_shader, "paletteSize",
                                 &renctx->data_palettesize_loc);

  mesh3d_shader_get_uniform_loc (renctx->dapu_shader, "TriangleInfo",
                                 &renctx->dapu_info_loc);
  mesh3d_shader_get_uniform_loc (renctx->dapu_shader, "RGBAColors",
                                 &renctx->dapu_colors_loc);
  mesh3d_shader_get_uniform_loc (renctx->dapu_shader, "Values",
                                 &renctx->dapu_values_loc);
  mesh3d_shader_get_uniform_loc (renctx->dapu_shader, "paletteSize",
                                 &renctx->dapu_palettesize_loc);
  mesh3d_shader_get_uniform_loc (renctx->dapu_shader, "dataBrightness",
                                 &renctx->dapu_brightness_loc);

  mesh3d_shader_get_uniform_loc (renctx->hud_shader, "digitTexture",
                                 &renctx->hud_digitex_loc);
  mesh3d_shader_get_uniform_loc (renctx->hud_shader, "digits",
                                 &renctx->hud_digits_loc);
  mesh3d_shader_get_uniform_loc (renctx->hud_shader, "line",
                                 &renctx->hud_line_loc);

  mesh3d_shader_get_uniform_loc (renctx->cut_shader, "TriangleInfo",
                                 &renctx->cut_info_loc);

  // Uniform Object Buffers
  mesh3d_shader_bind_uniform_buf (renctx->data_shader, "ColorBuf",
                                  BINDING_POINT_COLORBUF);
  mesh3d_shader_bind_uniform_buf (renctx->dapu_shader, "ColorBuf",
                                  BINDING_POINT_COLORBUF);

  mesh3d_shader_bind_uniform_buf (renctx->cpu_shader, "CustomColors",
                                  BINDING_POINT_CUSTOM_COLORS);
  mesh3d_shader_bind_uniform_buf (renctx->dapu_shader, "CustomColors",
                                  BINDING_POINT_CUSTOM_COLORS);
  mesh3d_shader_bind_uniform_buf (renctx->data_shader, "CustomColors",
                                  BINDING_POINT_CUSTOM_COLORS);
  mesh3d_shader_bind_uniform_buf (renctx->clipping_shader, "CustomColors",
                                  BINDING_POINT_CUSTOM_COLORS);
  mesh3d_shader_bind_uniform_buf (renctx->cut_shader, "CustomColors",
                                  BINDING_POINT_CUSTOM_COLORS);

  mesh3d_shader_bind_uniform_buf (renctx->cpu_shader, "Clipping",
                                  BINDING_POINT_CLIPPING);
  mesh3d_shader_bind_uniform_buf (renctx->data_shader, "Clipping",
                                  BINDING_POINT_CLIPPING);
  mesh3d_shader_bind_uniform_buf (renctx->dapu_shader, "Clipping",
                                  BINDING_POINT_CLIPPING);
  mesh3d_shader_bind_uniform_buf (renctx->picking_shader, "Clipping",
                                  BINDING_POINT_CLIPPING);
  mesh3d_shader_bind_uniform_buf (renctx->clipping_shader, "Clipping",
                                  BINDING_POINT_CLIPPING);
  mesh3d_shader_bind_uniform_buf (renctx->cut_shader, "Clipping",
                                  BINDING_POINT_CLIPPING);

  mesh3d_shader_bind_uniform_buf (renctx->hud_shader, "HudInfo",
                                  BINDING_POINT_HUDINFO);

  // Bind Matrices to all shaders
  mesh3d_shader_bind_uniform_buf (renctx->cpu_shader, "Matrices",
                                  BINDING_POINT_MATRICES);
  mesh3d_shader_bind_uniform_buf (renctx->data_shader, "Matrices",
                                  BINDING_POINT_MATRICES);
  mesh3d_shader_bind_uniform_buf (renctx->dapu_shader, "Matrices",
                                  BINDING_POINT_MATRICES);
  mesh3d_shader_bind_uniform_buf (renctx->picking_shader, "Matrices",
                                  BINDING_POINT_MATRICES);
  mesh3d_shader_bind_uniform_buf (renctx->hud_shader, "Matrices",
                                  BINDING_POINT_MATRICES);
  mesh3d_shader_bind_uniform_buf (renctx->clipping_shader, "Matrices",
                                  BINDING_POINT_MATRICES);
  mesh3d_shader_bind_uniform_buf (renctx->cut_shader, "Matrices",
                                  BINDING_POINT_MATRICES);

  if (ctx->picking_enabled)
    mesh3d_picking_init (ctx);

  if (ctx->hud_enabled)
    mesh3d_renderer_hud_init (ctx);

  if (ctx->clipping_enabled)
    mesh3d_zplane_init (ctx);
}

void mesh3d_renderer_set_mesh (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  // mesh is now known
  mesh3d_obj_t *mesh = ctx->mesh;

  // Initialize 'Matrices'
  mesh3d_renderer_mvp_init (ctx);

  mesh3d_switch_to_context (ctx);

  if (mesh->mesh_type == MESH3D_TYPE_VOLUME) {
    glEnable (GL_CULL_FACE);
    glCullFace (GL_BACK);
    glFrontFace (GL_CW);
  }

  // configure vertex attributes and misc buffers
  glGenVertexArrays (1, &renctx->VAO);
  glGenBuffers (1, &renctx->VBO);

  // bind the Vertex Array Object first, then bind and set vertex buffer(s)
  glBindVertexArray (renctx->VAO);
  glBindBuffer (GL_ARRAY_BUFFER, renctx->VBO);
  glBufferData (GL_ARRAY_BUFFER, mesh->nb_vertices * 3 * sizeof (GLfloat),
                mesh->vertices, GL_STATIC_DRAW); // Stored in GPU once for all
  // configure vertex attributes(s).
  glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof (float),
                         (void *)0);
  glEnableVertexAttribArray (0);

  glGenBuffers (1, &renctx->VBO_IND);
  glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, renctx->VBO_IND);
  glBufferData (GL_ELEMENT_ARRAY_BUFFER, mesh->nb_triangles * 3 * sizeof (int),
                mesh->triangles, GL_STATIC_DRAW); // Stored in GPU once for all

  // Texture Buffer Object containing per-triangle info
  glGenBuffers (1, &renctx->TBO_INFO);
  glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_INFO);
  glBufferData (GL_TEXTURE_BUFFER, mesh->nb_triangles * sizeof (int),
                mesh->triangle_info,
                GL_STATIC_DRAW); // Stored in GPU once for all
  glActiveTexture (GL_TEXTURE2);
  glGenTextures (1, &renctx->tex_tinfo);
  glBindTexture (GL_TEXTURE_BUFFER, renctx->tex_tinfo);
  glTexBuffer (GL_TEXTURE_BUFFER, GL_R32I, renctx->TBO_INFO);

  // bind uniform buffer objects to texture #2
  glProgramUniform1i (renctx->cpu_shader, renctx->cpu_info_loc, 2);
  glProgramUniform1i (renctx->data_shader, renctx->data_info_loc, 2);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_info_loc, 2);
  glProgramUniform1i (renctx->cut_shader, renctx->cut_info_loc, 2);
}

void mesh3d_renderer_use_cpu_palette (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  if (!mesh3d_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not defined");

  mesh3d_switch_to_context (ctx);

  // Texture Buffer Object containing RGBA colors
  glGenBuffers (1, &renctx->TBO_COL);
  glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_COL);
  glBufferData (GL_TEXTURE_BUFFER, ctx->mesh->nb_cells * sizeof (int), NULL,
                GL_DYNAMIC_DRAW); // To be refreshed regularly
  glActiveTexture (GL_TEXTURE1);
  glGenTextures (1, &renctx->tex_color);
  glBindTexture (GL_TEXTURE_BUFFER, renctx->tex_color);
  glTexBuffer (GL_TEXTURE_BUFFER, GL_R32I, renctx->TBO_COL);

  // bind uniform buffer object to texture #1
  glProgramUniform1i (renctx->cpu_shader, renctx->cpu_colors_loc, 1);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_colors_loc, 1);
}

void mesh3d_renderer_use_data_palette (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  if (!mesh3d_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  if (renctx->UBO_DATACOL != 0)
    exit_with_error ("data palette already set");

  mesh3d_switch_to_context (ctx);

  // Discrete Color Palette
  glGenBuffers (1, &renctx->UBO_DATACOL);
  glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_DATACOL);
  glBufferData (GL_UNIFORM_BUFFER,
                ctx->data_palette.max_colors * 4 * sizeof (float),
                ctx->data_palette.colors,
                GL_STATIC_DRAW); // Stored in GPU once for all
  glBindBufferBase (GL_UNIFORM_BUFFER, BINDING_POINT_COLORBUF,
                    renctx->UBO_DATACOL);

  glProgramUniform1i (renctx->data_shader, renctx->data_palettesize_loc,
                      ctx->data_palette.max_colors);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_palettesize_loc,
                      ctx->data_palette.max_colors);

  // Data brightness
  glProgramUniform1f (renctx->dapu_shader, renctx->dapu_brightness_loc, DEFAULT_DATA_BRIGHTNESS);

  // Texture Buffer Objects containing float data values [0.0 … 1.0]
  glGenBuffers (2, renctx->TBO_DATACOL);
  for (int i = 0; i < 2; i++) {
    glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_DATACOL[i]);
    glBufferData (GL_TEXTURE_BUFFER, ctx->mesh->nb_cells * sizeof (float), NULL,
                  GL_DYNAMIC_DRAW); // To be refreshed regularly
  }

  renctx->current_datacol = 0;
  glActiveTexture (GL_TEXTURE3);
  glGenTextures (1, &renctx->tex_datacol);
  glBindTexture (GL_TEXTURE_BUFFER, renctx->tex_datacol);
  glTexBuffer (GL_TEXTURE_BUFFER, GL_R32F,
               renctx->TBO_DATACOL[renctx->current_datacol]);

  // bind uniform buffer object to texture #3
  glProgramUniform1i (renctx->data_shader, renctx->data_values_loc, 3);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_values_loc, 3);
}

void mesh3d_set_data_brightness (mesh3d_ctx_t ctx, float brightness)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  mesh3d_switch_to_context (ctx);

  glProgramUniform1f (renctx->dapu_shader, renctx->dapu_brightness_loc, brightness);
}

void mesh3d_switch_data_color_buffer (mesh3d_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  renctx->current_datacol = 1 - renctx->current_datacol;
  mesh3d_switch_to_context (ctx);

  glActiveTexture (GL_TEXTURE3);
  glBindTexture (GL_TEXTURE_BUFFER, renctx->tex_datacol);
  glTexBuffer (GL_TEXTURE_BUFFER, GL_R32F,
               renctx->TBO_DATACOL[renctx->current_datacol]);
}

void mesh3d_get_sharable_buffer_ids (mesh3d_ctx_t ctx, int buffer_ids[])
{
  if (!mesh3d_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  mesh3d_switch_to_context (ctx);

  for (int i = 0; i < 2; i++)
    buffer_ids[i] = renctx->TBO_DATACOL[i];
}

void mesh3d_set_data_colors (mesh3d_ctx_t ctx, float *values)
{
  if (!mesh3d_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  mesh3d_render_ctx_t *renctx = ctx->render_ctx;

  mesh3d_switch_to_context (ctx);

  glBindBuffer (GL_TEXTURE_BUFFER,
                renctx->TBO_DATACOL[renctx->current_datacol]);
  glBufferSubData (GL_TEXTURE_BUFFER, 0, ctx->mesh->nb_cells * sizeof (float),
                   values);
}

int mesh3d_renderer_do_picking (mesh3d_ctx_t ctx, int x, int y)
{
  mesh3d_render_ctx_t *renctx = ctx->render_ctx;
  mesh3d_obj_t *mesh          = ctx->mesh;

  mesh3d_switch_to_context (ctx);

  glUseProgram (renctx->picking_shader);

  glBindFramebuffer (GL_DRAW_FRAMEBUFFER, renctx->FBO); // ON

  glDisable (GL_BLEND);
  glClearColor (0.0f, 0.0f, 0.0f, 1.0f);
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // draw triangles
  glDrawElements (GL_TRIANGLES, mesh->nb_triangles * 3, GL_UNSIGNED_INT, 0);

  glBindFramebuffer (GL_DRAW_FRAMEBUFFER, 0); // OFF

  // Read texture at position (mouse_x, mouse_y)
  glBindFramebuffer (GL_READ_FRAMEBUFFER, renctx->FBO);
  glReadBuffer (GL_COLOR_ATTACHMENT0);

  float pixel[1] = {0.0};

  glReadPixels (x, ctx->winh - y - 1, 1, 1, GL_RED, GL_FLOAT, pixel);

  glReadBuffer (GL_NONE);
  glBindFramebuffer (GL_READ_FRAMEBUFFER, 0);

  glEnable (GL_BLEND);

  int sel = (int)pixel[0] - 1; // triangle_no
  if (sel != -1)
    sel = mesh->triangle_info[sel] >> CELLNO_SHIFT; // triangle_no -> cell_no

  return sel;
}

void mesh3d_render (mesh3d_ctx_t ctx[], unsigned nb_ctx)
{
  for (int c = 0; c < nb_ctx; c++) {
    mesh3d_obj_t *mesh          = ctx[c]->mesh;
    mesh3d_render_ctx_t *renctx = ctx[c]->render_ctx;

    mesh3d_switch_to_context (ctx[c]);

    glClearColor (0.0f, 0.2f, 0.2f, 1.0f);
    glClear (GL_COLOR_BUFFER_BIT |
             GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

    if (mesh3d_palette_is_defined (&ctx[c]->cpu_palette)) {

      glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_COL);
      glBufferSubData (GL_TEXTURE_BUFFER, 0, mesh->nb_cells * sizeof (int),
                       ctx[c]->cpu_colors);

      if (mesh3d_palette_is_defined (&ctx[c]->data_palette))
        // Both CPU and Data colors are used
        glUseProgram (renctx->dapu_shader);
      else
        // CPU colors only
        glUseProgram (renctx->cpu_shader);

      glDrawElements (GL_TRIANGLES, mesh->nb_triangles * 3, GL_UNSIGNED_INT, 0);

    } else if (mesh3d_palette_is_defined (&ctx[c]->data_palette)) {
      // Data colors only
      glUseProgram (renctx->data_shader);

      glDrawElements (GL_TRIANGLES, mesh->nb_triangles * 3, GL_UNSIGNED_INT, 0);
    }

    if (ctx[c]->hud_enabled)
      // On-Screen Head Up Display
      mesh3d_hud_display (ctx[c]);

    if (ctx[c]->clipping_enabled && ctx[c]->clipping_active)
      // Show z clipping plane
      mesh3d_zplane_display (ctx[c]);

    SDL_GL_SwapWindow (ctx[c]->win);
  }
}

