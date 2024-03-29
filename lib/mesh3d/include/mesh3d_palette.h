#ifndef MESH3D_PALETTE_IS_DEF
#define MESH3D_PALETTE_IS_DEF

#define MAX_PALETTE 4096

typedef enum
{
  MESH3D_PALETTE_CUSTOM,
  MESH3D_PALETTE_LINEAR,
  MESH3D_PALETTE_HEAT,
  MESH3D_PALETTE_3GAUSS,
  MESH3D_PALETTE_LIFE,
  MESH3D_PALETTE_BARBIE_KEN,
  MESH3D_PALETTE_CHRISTMAS,
  MESH3D_PALETTE_YELLOW,
  MESH3D_PALETTE_RAINBOW,
} mesh3d_palette_name_t;

typedef struct
{
  mesh3d_palette_name_t name;
  unsigned max_colors;
  float *colors; // 4 floats per color
} mesh3d_palette_t;

void mesh3d_palette_init (mesh3d_palette_t *palette);
void mesh3d_palette_delete (mesh3d_palette_t *palette);

void mesh3d_palette_set_RGBA_passthrough (mesh3d_palette_t *palette);
void mesh3d_palette_set_raw (mesh3d_palette_t *palette, float *data,
                             unsigned size);
void mesh3d_palette_set_from_RGBAi (mesh3d_palette_t *palette,
                                    unsigned colors[], unsigned size);
void mesh3d_palette_set_predefined (mesh3d_palette_t *palette,
                                    mesh3d_palette_name_t name);

int mesh3d_palette_is_defined (mesh3d_palette_t *palette);


#endif
