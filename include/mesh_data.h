#ifndef MESH_DATA_IS_DEF
#define MESH_DATA_IS_DEF

#include "global.h"
#include "cppdefs.h"
#include "ezv.h"
#include "minmax.h"

#include <stdint.h>

extern unsigned NB_CELLS;
extern unsigned NB_PATCHES;
extern unsigned GPU_SIZE;
extern unsigned TILE;

extern float *RESTRICT mesh_data, *RESTRICT alt_mesh_data;

extern int *RESTRICT neighbors_soa;
extern int neighbor_soa_offset;

#define cur_data(c) (*(mesh_data + (c)))
#define next_data(c) (*(alt_mesh_data + (c)))

static inline void swap_data (void)
{
  float *tmp = mesh_data;

  mesh_data     = alt_mesh_data;
  alt_mesh_data = tmp;
}

void mesh_data_init (void);
void mesh_data_alloc (void);
void mesh_data_free (void);
void mesh_data_replicate (void);
void mesh_data_set_palette (float *data, unsigned size);
void mesh_data_set_palette_predefined (ezv_palette_name_t palette);
void mesh_data_set_default_palette_if_none_defined (void);
ezv_palette_name_t mesh_data_get_palette (void);
void mesh_data_init_huds (int show);
void mesh_data_refresh (unsigned iter);
void mesh_data_do_pick (void);
void mesh_data_dump_to_file (char *filename);
void mesh_data_save_thumbnail (unsigned iteration);
void mesh_data_build_neighbors_soa (unsigned round);

void mesh_data_reorder_partitions (int newpos[]);

extern mesh3d_obj_t easypap_mesh_desc;

static inline unsigned min_neighbors (void)
{
  return easypap_mesh_desc.min_neighbors;
}

static inline unsigned max_neighbors (void)
{
  return easypap_mesh_desc.max_neighbors;
}

static inline unsigned neighbor_start (int cell)
{
  return easypap_mesh_desc.index_first_neighbor[cell];
}

static inline unsigned neighbor_end (int cell)
{
  return easypap_mesh_desc.index_first_neighbor[cell + 1];
}

static inline unsigned nb_neighbors (int cell)
{
  return neighbor_end (cell) - neighbor_start (cell);
}

static inline unsigned nth_neighbor (int cell, int n)
{
  return easypap_mesh_desc.neighbors[neighbor_start (cell) + n];
}

static inline unsigned neighbor (int n)
{
  return easypap_mesh_desc.neighbors[n];
}

static inline unsigned patch_start (int p)
{
  return easypap_mesh_desc.patch_first_cell[p];
}

static inline unsigned patch_end (int p)
{
  return easypap_mesh_desc.patch_first_cell[p + 1];
}

static inline unsigned patch_size (int p)
{
  return patch_end (p) - patch_start (p);
}

static inline unsigned patch_neighbor_start (int p)
{
  return easypap_mesh_desc.index_first_patch_neighbor[p];
}

static inline unsigned patch_neighbor_end (int p)
{
  return easypap_mesh_desc.index_first_patch_neighbor[p + 1];
}

static inline unsigned patch_neighbor (int n)
{
  return easypap_mesh_desc.patch_neighbors[n];
}


extern unsigned picking_enabled;

#endif
