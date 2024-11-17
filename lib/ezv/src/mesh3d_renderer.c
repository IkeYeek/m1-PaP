#include <cglm/cglm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "error.h"
#include "ezv_ctx.h"
#include "ezv_hud.h"
#include "ezv_mesh3d_object.h"
#include "ezv_sdl_gl.h"
#include "ezv_shader.h"
#include "ezv_textures.h"
#include "mesh3d_renderer.h"
#include "stb_image.h"

// #define WHITE_BACKGROUND 1

static const float DEFAULT_DATA_BRIGHTNESS = 0.8f;

static const float NEARZ = 0.2f;
static const float FARZ  = 5.0f;

#define INITIAL_TRANSLATE_VAL -1.5f
#define INITIAL_ROTATE_Y -90.0f
#define INITIAL_ROTATE_X +15.0f

typedef struct render_ctx_s
{
  GLuint UBO_DATACOL;
  GLuint UBO_MAT, UBO_CUST, UBO_CLIP;
  GLuint VBO; // Vertex Buffer Object (contains vertices)
  GLuint VAO; // Vertex Array Object
  GLuint VBO_ZPLANE, VAO_ZPLANE, EBO_ZPLANE; // for zplane clipping
  GLuint VBO_IND;  // Vertex Buffer Object containing triangles (i.e. 3-tuples
                   // indexing vertices)
  GLuint TBO_INFO; // Texture Buffer Object containing per-triangle info
                   // (cellno, drop, edgen)
  GLuint TBO_COL;  // Texture Buffer Object containing per-cell indexes to cpu
                   // color palette
  GLuint TBO_DATACOL;
  GLuint tex_tinfo;   // GL Texture associated to TBO_INFO
  GLuint tex_color;   // TBO_COL
  GLuint tex_datacol; // TBO_DATACOL
  GLuint FBO;
  GLuint pickingTexture, depthTexture, imgTexture;
  GLuint cpu_shader, picking_shader, data_shader, dapu_shader, clipping_shader,
      cut_shader;
  GLuint cpu_colors_loc, cpu_info_loc;
  GLuint data_values_loc, data_info_loc, data_palettesize_loc;
  GLuint dapu_colors_loc, dapu_info_loc, dapu_values_loc, dapu_palettesize_loc,
      dapu_brightness_loc;
  GLuint cut_info_loc;
} mesh3d_render_ctx_t;

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

static mat4 projection, unclipped_proj;

// Normalization operations translate and rescale the object to fit into a
// normalized bounding-box centered in (0,0,0)
static vec3 norm_scale, norm_translate;

static void set_transformed_zplane (void)
{
  // We must pass the projected z value of the zplane…
  vec4 center = {0.0, 0.0, Clipping.clipping_zcoord, 1.0};
  vec4 proj;

  glm_mat4_mulv (Matrices.vp_unclipped, center, proj);

  Clipping.clipping_zproj = proj[3];
}

void mesh3d_reset_view (ezv_ctx_t ctx[], unsigned nb_ctx)
{
  rotate_x    = INITIAL_ROTATE_X;
  rotate_y    = INITIAL_ROTATE_Y;
  translate_z = INITIAL_TRANSLATE_VAL;

  Clipping.clipping_zcoord = 0.0f;

  mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, 0.0f);
}

static void mesh3d_renderer_mvp_init (ezv_ctx_t ctx)
{
  static int done             = 0;
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);

  // Create matrices and vector once
  if (!done) {
    bbox_t *bbox = &mesh->bbox;
    GLfloat msize;
    vec3 dim;
    mat4 model, view;

    if (!mesh->bbox_set)
      mesh3d_obj_compute_bounding_box (mesh);

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

    glm_ortho (0.0f, (float)ctx->winw, (float)ctx->winh, 0.0f, -2.0f, 2.0f,
               Matrices.ortho);

    if (mesh->mesh_type == MESH3D_TYPE_SURFACE)
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

void mesh3d_renderer_mvp_update (ezv_ctx_t ctx[], unsigned nb_ctx, float dx,
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
    mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx[c]);

    ezv_switch_to_context (ctx[c]);

    glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_MAT);
    glBufferSubData (GL_UNIFORM_BUFFER, 0, sizeof (Matrices), &Matrices);

    glBindBuffer (GL_UNIFORM_BUFFER, renctx->UBO_CLIP);
    glBufferSubData (GL_UNIFORM_BUFFER, 0, sizeof (Clipping), &Clipping);
  }
}

void mesh3d_renderer_zplane_update (ezv_ctx_t ctx[], unsigned nb_ctx, float dz)
{
  Clipping.clipping_zcoord += dz;
  Clipping.clipping_active = ctx[0]->clipping_active;

  mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, 0.0f);
}

static void mesh3d_zplane_init (ezv_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);

  float vertices[] = {
      // 2D positions
      -.65f, .65f * (float)ctx->winh / (float)ctx->winw,  //
      -.65f, -.65f * (float)ctx->winh / (float)ctx->winw, //
      .65f,  -.65f * (float)ctx->winh / (float)ctx->winw, //
      .65f,  .65f * (float)ctx->winh / (float)ctx->winw,
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

static void mesh3d_zplane_display (ezv_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);

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

static void mesh3d_picking_init (ezv_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);

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
void mesh3d_renderer_init (ezv_ctx_t ctx)
{
  ezv_switch_to_context (ctx);

  // configure global opengl state
  // -----------------------------
  glEnable (GL_DEPTH_TEST);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable (GL_BLEND);
  glEnable (GL_LINE_SMOOTH);
  glLineWidth (.75);

  // Allocate render_ctx
  mesh3d_render_ctx_t *renctx = malloc (sizeof (mesh3d_render_ctx_t));
  renctx->UBO_DATACOL         = 0;
  renctx->UBO_MAT             = 0;
  renctx->UBO_CUST            = 0;
  renctx->UBO_CLIP            = 0;
  renctx->VBO                 = 0;
  renctx->VAO                 = 0;
  renctx->VBO_ZPLANE          = 0;
  renctx->VAO_ZPLANE          = 0;
  renctx->EBO_ZPLANE          = 0;
  renctx->VBO_IND             = 0;
  renctx->TBO_INFO            = 0;
  renctx->TBO_COL             = 0;
  renctx->TBO_DATACOL         = 0;
  renctx->tex_tinfo           = 0;
  renctx->tex_color           = 0;
  renctx->depthTexture        = 0;
  renctx->pickingTexture      = 0;
  renctx->imgTexture          = 0;

  ezv_mesh3d_set_renderer (ctx, renctx);

  // compile shaders and build program
  renctx->cpu_shader  = ezv_shader_create ("mesh3d/generic.vs", "mesh3d/cpu.gs",
                                           "mesh3d/generic.fs");
  renctx->data_shader = ezv_shader_create (
      "mesh3d/generic.vs", "mesh3d/data.gs", "mesh3d/generic.fs");
  renctx->dapu_shader = ezv_shader_create (
      "mesh3d/generic.vs", "mesh3d/cpu_data.gs", "mesh3d/generic.fs");
  renctx->picking_shader = ezv_shader_create (
      "mesh3d/generic.vs", "mesh3d/generic.gs", "mesh3d/picking.fs");
  renctx->clipping_shader =
      ezv_shader_create ("mesh3d/clipping.vs", NULL, "mesh3d/cut.fs");
  renctx->cut_shader =
      ezv_shader_create ("mesh3d/cut.vs", "mesh3d/cut.gs", "mesh3d/cut.fs");

  // Uniform parameters
  ezv_shader_get_uniform_loc (renctx->cpu_shader, "TriangleInfo",
                              &renctx->cpu_info_loc);
  ezv_shader_get_uniform_loc (renctx->cpu_shader, "RGBAColors",
                              &renctx->cpu_colors_loc);

  ezv_shader_get_uniform_loc (renctx->data_shader, "TriangleInfo",
                              &renctx->data_info_loc);
  ezv_shader_get_uniform_loc (renctx->data_shader, "Values",
                              &renctx->data_values_loc);
  ezv_shader_get_uniform_loc (renctx->data_shader, "paletteSize",
                              &renctx->data_palettesize_loc);

  ezv_shader_get_uniform_loc (renctx->dapu_shader, "TriangleInfo",
                              &renctx->dapu_info_loc);
  ezv_shader_get_uniform_loc (renctx->dapu_shader, "RGBAColors",
                              &renctx->dapu_colors_loc);
  ezv_shader_get_uniform_loc (renctx->dapu_shader, "Values",
                              &renctx->dapu_values_loc);
  ezv_shader_get_uniform_loc (renctx->dapu_shader, "paletteSize",
                              &renctx->dapu_palettesize_loc);
  ezv_shader_get_uniform_loc (renctx->dapu_shader, "dataBrightness",
                              &renctx->dapu_brightness_loc);

  ezv_shader_get_uniform_loc (renctx->cut_shader, "TriangleInfo",
                              &renctx->cut_info_loc);

  // Uniform Object Buffers
  ezv_shader_bind_uniform_buf (renctx->data_shader, "ColorBuf",
                               BINDING_POINT_COLORBUF);
  ezv_shader_bind_uniform_buf (renctx->dapu_shader, "ColorBuf",
                               BINDING_POINT_COLORBUF);

  ezv_shader_bind_uniform_buf (renctx->cpu_shader, "CustomColors",
                               BINDING_POINT_CUSTOM_COLORS);
  ezv_shader_bind_uniform_buf (renctx->dapu_shader, "CustomColors",
                               BINDING_POINT_CUSTOM_COLORS);
  ezv_shader_bind_uniform_buf (renctx->data_shader, "CustomColors",
                               BINDING_POINT_CUSTOM_COLORS);
  ezv_shader_bind_uniform_buf (renctx->clipping_shader, "CustomColors",
                               BINDING_POINT_CUSTOM_COLORS);
  ezv_shader_bind_uniform_buf (renctx->cut_shader, "CustomColors",
                               BINDING_POINT_CUSTOM_COLORS);

  ezv_shader_bind_uniform_buf (renctx->cpu_shader, "Clipping",
                               BINDING_POINT_CLIPPING);
  ezv_shader_bind_uniform_buf (renctx->data_shader, "Clipping",
                               BINDING_POINT_CLIPPING);
  ezv_shader_bind_uniform_buf (renctx->dapu_shader, "Clipping",
                               BINDING_POINT_CLIPPING);
  ezv_shader_bind_uniform_buf (renctx->picking_shader, "Clipping",
                               BINDING_POINT_CLIPPING);
  ezv_shader_bind_uniform_buf (renctx->clipping_shader, "Clipping",
                               BINDING_POINT_CLIPPING);
  ezv_shader_bind_uniform_buf (renctx->cut_shader, "Clipping",
                               BINDING_POINT_CLIPPING);

  // Bind Matrices to all shaders
  ezv_shader_bind_uniform_buf (renctx->cpu_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->data_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->dapu_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->picking_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->clipping_shader, "Matrices",
                               BINDING_POINT_MATRICES);
  ezv_shader_bind_uniform_buf (renctx->cut_shader, "Matrices",
                               BINDING_POINT_MATRICES);

  if (ctx->picking_enabled)
    mesh3d_picking_init (ctx);

  if (ctx->clipping_enabled)
    mesh3d_zplane_init (ctx);
}

void mesh3d_renderer_set_mesh (ezv_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);

  ezv_switch_to_context (ctx);

  // Initialize 'Matrices'
  mesh3d_renderer_mvp_init (ctx);

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
  glActiveTexture (GL_TEXTURE0 + EZV_INFO_TEXTURE_NUM);
  glGenTextures (1, &renctx->tex_tinfo);
  glBindTexture (GL_TEXTURE_BUFFER, renctx->tex_tinfo);
  glTexBuffer (GL_TEXTURE_BUFFER, GL_R32I, renctx->TBO_INFO);

  // bind uniform buffer objects to texture #2
  glProgramUniform1i (renctx->cpu_shader, renctx->cpu_info_loc,
                      EZV_INFO_TEXTURE_NUM);
  glProgramUniform1i (renctx->data_shader, renctx->data_info_loc,
                      EZV_INFO_TEXTURE_NUM);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_info_loc,
                      EZV_INFO_TEXTURE_NUM);
  glProgramUniform1i (renctx->cut_shader, renctx->cut_info_loc,
                      EZV_INFO_TEXTURE_NUM);
}

void ezv_mesh3d_refresh_mesh (ezv_ctx_t ctx[], unsigned nb_ctx)
{
  for (int c = 0; c < nb_ctx; c++) {
    mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx[c]);
    mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx[c]);

    ezv_switch_to_context (ctx[c]);

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, renctx->VBO_IND);
    glBufferSubData (GL_ELEMENT_ARRAY_BUFFER, 0,
                     mesh->nb_triangles * 3 * sizeof (int), mesh->triangles);

    // Texture Buffer Object containing per-triangle info
    glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_INFO);
    glBufferSubData (GL_TEXTURE_BUFFER, 0, mesh->nb_triangles * sizeof (int),
                     mesh->triangle_info);
  }
}

void mesh3d_renderer_use_cpu_palette (ezv_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);

  if (!ezv_palette_is_defined (&ctx->cpu_palette))
    exit_with_error ("CPU palette not defined");

  ezv_switch_to_context (ctx);

  // Texture Buffer Object containing RGBA colors
  glGenBuffers (1, &renctx->TBO_COL);
  glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_COL);
  glBufferData (GL_TEXTURE_BUFFER, mesh->nb_cells * sizeof (int), NULL,
                GL_DYNAMIC_DRAW); // To be refreshed regularly

  glActiveTexture (GL_TEXTURE0 + EZV_CPU_TEXTURE_NUM);
  glGenTextures (1, &renctx->tex_color);
  glBindTexture (GL_TEXTURE_BUFFER, renctx->tex_color);
  glTexBuffer (GL_TEXTURE_BUFFER, GL_R32I, renctx->TBO_COL);

  // bind uniform buffer object to texture #1
  glProgramUniform1i (renctx->cpu_shader, renctx->cpu_colors_loc,
                      EZV_CPU_TEXTURE_NUM);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_colors_loc,
                      EZV_CPU_TEXTURE_NUM);
}

void mesh3d_renderer_use_data_palette (ezv_ctx_t ctx)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);

  if (!ezv_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  if (ctx->data_palette.name == EZV_PALETTE_RGBA_PASSTHROUGH)
    exit_with_error ("RGBA passthrough not supported yet");

  if (renctx->UBO_DATACOL != 0)
    exit_with_error ("data palette already set");

  ezv_switch_to_context (ctx);

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
  glProgramUniform1f (renctx->dapu_shader, renctx->dapu_brightness_loc,
                      DEFAULT_DATA_BRIGHTNESS);

  // Texture Buffer Objects containing float data values [0.0 … 1.0]
  glGenBuffers (1, &renctx->TBO_DATACOL);
  glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_DATACOL);
  glBufferData (GL_TEXTURE_BUFFER, mesh->nb_cells * sizeof (float), NULL,
                GL_DYNAMIC_DRAW); // To be refreshed regularly

  glActiveTexture (GL_TEXTURE0 + EZV_DATA_TEXTURE_NUM);
  glGenTextures (1, &renctx->tex_datacol);
  glBindTexture (GL_TEXTURE_BUFFER, renctx->tex_datacol);
  glTexBuffer (GL_TEXTURE_BUFFER, GL_R32F, renctx->TBO_DATACOL);

  // bind uniform buffer object to texture #3
  glProgramUniform1i (renctx->data_shader, renctx->data_values_loc,
                      EZV_DATA_TEXTURE_NUM);
  glProgramUniform1i (renctx->dapu_shader, renctx->dapu_values_loc,
                      EZV_DATA_TEXTURE_NUM);
}

void mesh3d_set_data_brightness (ezv_ctx_t ctx, float brightness)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);

  ezv_switch_to_context (ctx);

  glProgramUniform1f (renctx->dapu_shader, renctx->dapu_brightness_loc,
                      brightness);
}

void mesh3d_get_shareable_buffer_ids (ezv_ctx_t ctx, int buffer_ids[])
{
  if (!ezv_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);

  ezv_switch_to_context (ctx);

  buffer_ids[0] = renctx->TBO_DATACOL;
}

void mesh3d_set_data_colors (ezv_ctx_t ctx, void *values)
{
  if (!ezv_palette_is_defined (&ctx->data_palette))
    exit_with_error ("Data palette not defined");

  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);

  ezv_switch_to_context (ctx);

  glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_DATACOL);
  glBufferSubData (GL_TEXTURE_BUFFER, 0, mesh->nb_cells * sizeof (float),
                   values);
}

int mesh3d_renderer_do_picking (ezv_ctx_t ctx, int x, int y)
{
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);

  ezv_switch_to_context (ctx);

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

void mesh3d_render (ezv_ctx_t ctx)
{
  mesh3d_obj_t *mesh          = ezv_mesh3d_mesh (ctx);
  mesh3d_render_ctx_t *renctx = ezv_mesh3d_renderer (ctx);

  ezv_switch_to_context (ctx);

#ifdef WHITE_BACKGROUND
  glClearColor (1.0f, 1.f, 1.0f, 1.0f);
#else
  glClearColor (0.0f, 0.2f, 0.2f, 1.0f);
#endif
  glClear (GL_COLOR_BUFFER_BIT |
           GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!

  if (ezv_palette_is_defined (&ctx->cpu_palette)) {

    glBindBuffer (GL_TEXTURE_BUFFER, renctx->TBO_COL);
    glBufferSubData (GL_TEXTURE_BUFFER, 0, mesh->nb_cells * sizeof (int),
                     ctx->cpu_colors);

    if (ezv_palette_is_defined (&ctx->data_palette))
      // Both CPU and Data colors are used
      glUseProgram (renctx->dapu_shader);
    else
      // CPU colors only
      glUseProgram (renctx->cpu_shader);

    glDrawElements (GL_TRIANGLES, mesh->nb_triangles * 3, GL_UNSIGNED_INT, 0);

  } else if (ezv_palette_is_defined (&ctx->data_palette)) {
    // Data colors only
    glUseProgram (renctx->data_shader);

    glDrawElements (GL_TRIANGLES, mesh->nb_triangles * 3, GL_UNSIGNED_INT, 0);
  }

  if (ctx->hud_enabled) {
    // On-Screen Head Up Display
    ezv_hud_display (ctx);
    glBindVertexArray (renctx->VAO);
  }

  if (ctx->clipping_enabled && ctx->clipping_active)
    // Show z clipping plane
    mesh3d_zplane_display (ctx);

  SDL_GL_SwapWindow (ctx->win);
}

int mesh3d_renderer_zoom (ezv_ctx_t ctx[], unsigned nb_ctx, unsigned shift_mod,
                          unsigned in)
{
  if (shift_mod) {
    // move clipping plane forward
    if (ctx[0]->clipping_active) {
      mesh3d_renderer_zplane_update (ctx, nb_ctx, in ? +0.01f : -0.01f);
      return 1;
    }
  } else {
    // Zoom in
    mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.0f, 0.0f, in ? +.02f : -.02f);
    return 1;
  }
  return 0;
}

int mesh3d_renderer_motion (ezv_ctx_t ctx[], unsigned nb_ctx, int dx, int dy,
                            unsigned wheel)
{
  if (wheel)
    mesh3d_renderer_mvp_update (ctx, nb_ctx, 1.5f * dy, 1.5f * dx, 0.0f);
  else
    mesh3d_renderer_mvp_update (ctx, nb_ctx, 0.2f * dy, 0.2f * dx, 0.0f);

  return 1;
}
