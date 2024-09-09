#ifndef MESH_MGPU_ISDEF
#define MESH_MGPU_ISDEF

#ifdef ENABLE_OPENCL

#include <unistd.h>

#include "gpu.h"

void mesh_mgpu_config (unsigned halo_width);
void mesh_mgpu_init (void);
void mesh_mgpu_exchg_borders (void);
void mesh_mgpu_send_initial_data (void);
void mesh_mgpu_refresh_img (void);
void mesh_mgpu_debug (int cell);
void mesh_mgpu_overlay (int cell);

int mesh_mgpu_legacy_cells (int gpu);
int mesh_mgpu_offset_cells (int gpu);
int mesh_mgpu_cells_to_compute (int gpu);
int mesh_mgpu_nb_threads (int gpu);

#ifdef ENABLE_OPENCL

cl_mem mesh_mgpu_get_soa_buffer (int gpu);

#endif

#endif

#endif
