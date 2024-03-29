#ifndef MESH3D_OBJ_IS_DEF
#define MESH3D_OBJ_IS_DEF

#include <cglm/cglm.h>

typedef enum
{
  MESH3D_TYPE_SURFACE,
  MESH3D_TYPE_VOLUME
} mesh3d_type_t;

// Per-triangle information (triangle_info field):
// bit  [0] = isInner
// bits [1..3] = <edge0, edge1, edge2>
// bits [4..6] = <frontier0, frontier1, frontier2>
// bits [7..31] = cell_no (limited to 2^25 = 32M cells)
#define ISINNER (1U << 0)
#define EDGE0 (1U << 1)
#define EDGE1 (1U << 2)
#define EDGE2 (1U << 3)
#define FRONTIER_SHIFT 4U
#define FRONTIER_MASK  7U
#define FRONTIER(n) (1U << (FRONTIER_SHIFT + n))
#define CELLNO_SHIFT 7U

typedef struct
{
  vec3 min, max;
} bbox_t;

#define MESH3D_PART_USE_SCOTCH     1
#define MESH3D_PART_SHOW_FRONTIERS 2

typedef struct
{
  bbox_t bbox;
  mesh3d_type_t mesh_type;
  float *vertices;
  unsigned nb_vertices;
  unsigned *triangles;
  unsigned nb_triangles;
  unsigned *cells;
  unsigned *triangle_info;
  unsigned nb_cells;
  unsigned min_neighbors, max_neighbors;
  unsigned total_neighbors;
  unsigned *neighbors;
  unsigned *index_first_neighbor;
  unsigned nb_patches;
  unsigned *patch_first_cell;
} mesh3d_obj_t;

void mesh3d_obj_init (mesh3d_obj_t *mesh);
void mesh3d_obj_build_default (mesh3d_obj_t *mesh);

void mesh3d_obj_build_cube (mesh3d_obj_t *mesh, unsigned group_size);
void mesh3d_obj_build_torus (mesh3d_obj_t *mesh, unsigned group_size);
void mesh3d_obj_build_torus_surface (mesh3d_obj_t *mesh, unsigned group_size);
void mesh3d_obj_build_wall (mesh3d_obj_t *mesh, unsigned size);

void mesh3d_obj_build_cube_volume (mesh3d_obj_t *mesh, unsigned size);
void mesh3d_obj_build_torus_volume (mesh3d_obj_t *mesh, unsigned size_x,
                                    unsigned size_y, unsigned size_z);
void mesh3d_obj_build_cylinder_volume (mesh3d_obj_t *mesh, unsigned size_x,
                                       unsigned size_y);

void mesh3d_obj_load (const char *filename, mesh3d_obj_t *mesh);
void mesh3d_obj_store (const char *filename, mesh3d_obj_t *mesh,
                       int with_patches);

void mesh3d_obj_partition (mesh3d_obj_t *mesh, unsigned nbpart,
                           int flag);
int mesh3d_obj_get_patch_of_cell (mesh3d_obj_t *mesh, unsigned cell);
void mesh3d_obj_get_bbox_of_cell (mesh3d_obj_t *mesh, unsigned cell, bbox_t *box);
void mesh3d_obj_get_barycenter (mesh3d_obj_t *mesh, unsigned cell,
                                float *bx, float *by, float *bz);

#endif
