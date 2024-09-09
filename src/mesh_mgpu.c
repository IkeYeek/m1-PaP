#ifdef ENABLE_OPENCL

#include "mesh_mgpu.h"
#include "ezp_ctx.h"
#include "mesh_data.h"

typedef struct
{
  unsigned nb_cells;
  unsigned inner_halo;
  unsigned *halo_size;
  unsigned part_offset;
  unsigned outgoing_size;
  unsigned *outgoing_offsets;
  unsigned incoming_size;
  unsigned *incoming_offsets;
} mgpu_part_info_t;

typedef struct
{
  float *incoming_values;
  unsigned *outgoing_index;
  float *outgoing_values;
  cl_mem outgoing_index_buffer, outgoing_values_buffer;
  int *neighbor_soa;
  unsigned neighbor_offset;
  cl_mem soa_buffer;
  cl_kernel gather_kernel;
} mgpu_gpu_data_t;

static mgpu_part_info_t *mesh_mgpu_info = NULL;

static unsigned halo = 1;

mgpu_gpu_data_t gpu_data[MAX_NB_GPU] = {
    {NULL, NULL, NULL, 0, 0, NULL, 0, 0},
    {NULL, NULL, NULL, 0, 0, NULL, 0, 0},
};

static int mgpu_hud = -1;

int mesh_mgpu_legacy_cells (int gpu)
{
  return mesh_mgpu_info[gpu].nb_cells;
}

int mesh_mgpu_offset_cells (int gpu)
{
  return mesh_mgpu_info[gpu].part_offset;
}

int mesh_mgpu_cells_to_compute (int gpu)
{
  return mesh_mgpu_info[gpu].nb_cells + mesh_mgpu_info[gpu].incoming_size -
         mesh_mgpu_info[gpu].halo_size[halo - 1];
}

int mesh_mgpu_nb_threads (int gpu)
{
  return ROUND_TO_MULTIPLE (
      mesh_mgpu_info[gpu].nb_cells + mesh_mgpu_info[gpu].incoming_size, TILE);
}

#ifdef ENABLE_OPENCL

cl_mem mesh_mgpu_get_soa_buffer (int gpu)
{
  return gpu_data[gpu].soa_buffer;
}

#endif

#define index2d(array, y, x) (*(&array[(y) * easypap_mesh_desc.nb_metap + (x)]))

#define border_size(p1, p2) index2d (border_mat, p1, p2)
#define fill_ptr(p1, p2) index2d (prefix, p1, p2)

static void mesh_mgpu_display_stats (void)
{
  for (int i = 0; i < easypap_mesh_desc.nb_metap; i++) {
    PRINT_DEBUG ('m', "MP%d stores %d cells\n", i, mesh_mgpu_info[i].nb_cells);
    PRINT_DEBUG (
        'm', "MP%d has a total of %d outgoing and %d incoming border cells\n",
        i, mesh_mgpu_info[i].outgoing_size, mesh_mgpu_info[i].incoming_size);
    for (int h = 0; h < halo; h++)
      PRINT_DEBUG ('m', "MP%d level-%d halo size: %d\n", i, h,
                   mesh_mgpu_info[i].halo_size[h]);
    PRINT_DEBUG ('m', "MP%d has %d cells to compute\n", i,
                 mesh_mgpu_cells_to_compute (i));
  }
}

static void mesh_mgpu_build_info (void)
{
  // alloc part info
  mesh_mgpu_info =
      calloc (easypap_mesh_desc.nb_metap, sizeof (mgpu_part_info_t));
  for (int p = 0; p < easypap_mesh_desc.nb_metap; p++) {
    mesh_mgpu_info[p].outgoing_offsets =
        calloc (easypap_mesh_desc.nb_metap + 1, sizeof (unsigned));
    mesh_mgpu_info[p].incoming_offsets =
        calloc (easypap_mesh_desc.nb_metap + 1, sizeof (unsigned));
    mesh_mgpu_info[p].halo_size = calloc (halo, sizeof (unsigned));
  }

  // build an array storing the meta-partition of cell
  unsigned *tmp_metap_of_cell =
      malloc (easypap_mesh_desc.nb_cells * sizeof (unsigned));

  for (int mp = 0; mp < easypap_mesh_desc.nb_metap; mp++)
    for (int p = easypap_mesh_desc.metap_first_patch[mp];
         p < easypap_mesh_desc.metap_first_patch[mp + 1]; p++)
      for (int c = easypap_mesh_desc.patch_first_cell[p];
           c < easypap_mesh_desc.patch_first_cell[p + 1]; c++) {
        tmp_metap_of_cell[c] = mp;
        mesh_mgpu_info[mp].nb_cells++;
      }

  // compute prefix sum
  for (int mp = 1; mp < easypap_mesh_desc.nb_metap; mp++)
    mesh_mgpu_info[mp].part_offset =
        mesh_mgpu_info[mp - 1].part_offset + mesh_mgpu_info[mp - 1].nb_cells;

  // alloc border desc
  unsigned *border_mat =
      calloc (easypap_mesh_desc.nb_metap * easypap_mesh_desc.nb_metap,
              sizeof (unsigned));
  uint8_t *border_info = calloc (easypap_mesh_desc.nb_cells, sizeof (uint8_t));

  // First pass : level-0 halo
  // Compute outgoing and incoming buffer sizes
  for (int c = 0; c < easypap_mesh_desc.nb_cells; c++) {
    int mp = tmp_metap_of_cell[c];
    for (int ni = easypap_mesh_desc.index_first_neighbor[c];
         ni < easypap_mesh_desc.index_first_neighbor[c + 1]; ni++) {
      int n_mp = tmp_metap_of_cell[easypap_mesh_desc.neighbors[ni]];
      if (n_mp != mp) {
        uint32_t mask = 1U << n_mp;
        if (!(border_info[c] & mask)) {
          border_info[c] |= mask;
          border_size (mp, n_mp)++;
          mesh_mgpu_info[mp].outgoing_size++;
          mesh_mgpu_info[n_mp].incoming_size++;
          mesh_mgpu_info[n_mp].halo_size[0]++;
        }
      }
    }
  }

  uint8_t *binfo = NULL;

  if (halo > 1)
    binfo = calloc (easypap_mesh_desc.nb_cells, sizeof (uint8_t));

  // Remaining halo-1 passes
  for (int h = 1; h < halo; h++) {

    // Inner-halo contains all but last added cells
    for (int mp = 0; mp < easypap_mesh_desc.nb_metap; mp++)
      mesh_mgpu_info[mp].inner_halo = mesh_mgpu_info[mp].incoming_size;

    bzero (binfo, easypap_mesh_desc.nb_cells * sizeof (uint8_t));

    for (int c = 0; c < easypap_mesh_desc.nb_cells; c++) {
      int mp = tmp_metap_of_cell[c];
      for (int ni = easypap_mesh_desc.index_first_neighbor[c];
           ni < easypap_mesh_desc.index_first_neighbor[c + 1]; ni++) {
        // Try each possible meta partition
        for (int n_mp = 0; n_mp < easypap_mesh_desc.nb_metap; n_mp++) {
          if (n_mp != mp) {
            uint32_t mask = 1U << n_mp;
            // And see if neighbor cell belongs to n_mp's halo
            // If so, we now belong to n_mp's halo too
            if (border_info[easypap_mesh_desc.neighbors[ni]] &&
                !(border_info[c] & mask) && !(binfo[c] & mask)) {
#if 0
              PRINT_DEBUG (
                  'm', "Found a new cell belonging to MP%d's halo: cell %d\n",
                  n_mp, c);
#endif
              binfo[c] |= mask;
              border_size (mp, n_mp)++;
              mesh_mgpu_info[mp].outgoing_size++;
              mesh_mgpu_info[n_mp].incoming_size++;
              mesh_mgpu_info[n_mp].halo_size[h]++;
            }
          }
        }
      }
    }
    // we now have to merge binfo into border_info
    for (int c = 0; c < easypap_mesh_desc.nb_cells; c++)
      border_info[c] |= binfo[c];
  }

  // Now we can compute offsets of each set of neighbors
  for (int p = 0; p < easypap_mesh_desc.nb_metap; p++) {
    mgpu_part_info_t *pi    = mesh_mgpu_info + p;
    pi->outgoing_offsets[0] = 0;
    for (int i = 1; i <= easypap_mesh_desc.nb_metap; i++)
      pi->outgoing_offsets[i] =
          pi->outgoing_offsets[i - 1] + border_size (p, i - 1);
    pi->incoming_offsets[0] = pi->nb_cells;
    for (int i = 1; i <= easypap_mesh_desc.nb_metap; i++)
      pi->incoming_offsets[i] =
          pi->incoming_offsets[i - 1] + border_size (i - 1, p);
  }

  free (border_mat);
  border_mat = NULL;

  unsigned *prefix = malloc (easypap_mesh_desc.nb_metap *
                             easypap_mesh_desc.nb_metap * sizeof (unsigned));

  // Prepare fill_ptr matrix
  for (int mp = 0; mp < easypap_mesh_desc.nb_metap; mp++) {

    gpu_data[mp].outgoing_index =
        malloc (mesh_mgpu_info[mp].outgoing_size * sizeof (unsigned));

    for (int n_mp = 0; n_mp < easypap_mesh_desc.nb_metap; n_mp++)
      fill_ptr (mp, n_mp) = mesh_mgpu_info[mp].outgoing_offsets[n_mp];
  }

  // level-0 halo
  for (int c = 0; c < easypap_mesh_desc.nb_cells; c++) {
    int mp         = tmp_metap_of_cell[c];
    border_info[c] = 0;
    for (int ni = easypap_mesh_desc.index_first_neighbor[c];
         ni < easypap_mesh_desc.index_first_neighbor[c + 1]; ni++) {
      int n_mp = tmp_metap_of_cell[easypap_mesh_desc.neighbors[ni]];
      if (n_mp != mp) {
        uint32_t mask = 1U << n_mp;
        if (!(border_info[c] & mask)) {
          border_info[c] |= mask;
          gpu_data[mp].outgoing_index[fill_ptr (mp, n_mp)++] =
              c - mesh_mgpu_info[mp].part_offset;
#if 0
          PRINT_DEBUG ('m', "MP%d: cell %d -> outgoing[%d]\n", mp, c,
                       fill_ptr (mp, n_mp) - 1);
#endif
        }
      }
    }
  }

  // Remaining level-[1..n] halos
  for (int h = 1; h < halo; h++) {
    bzero (binfo, easypap_mesh_desc.nb_cells * sizeof (uint8_t));

    for (int c = 0; c < easypap_mesh_desc.nb_cells; c++) {
      int mp = tmp_metap_of_cell[c];
      for (int ni = easypap_mesh_desc.index_first_neighbor[c];
           ni < easypap_mesh_desc.index_first_neighbor[c + 1]; ni++) {
        // Try each possible meta partition
        for (int n_mp = 0; n_mp < easypap_mesh_desc.nb_metap; n_mp++) {
          if (n_mp != mp) {
            uint32_t mask = 1U << n_mp;
            // And see if neighbor cell belongs to n_mp's halo
            // If so, we now belong to n_mp's halo too
            if (border_info[easypap_mesh_desc.neighbors[ni]] &&
                !(border_info[c] & mask) && !(binfo[c] & mask)) {
              binfo[c] |= mask;
              gpu_data[mp].outgoing_index[fill_ptr (mp, n_mp)++] =
                  c - mesh_mgpu_info[mp].part_offset;
#if 0
              PRINT_DEBUG ('m', "MP%d: cell %d -> outgoing[%d]\n", mp, c,
                           fill_ptr (mp, n_mp) - 1);
#endif
            }
          }
        }
      }
    }
    // we now have to merge binfo into border_info
    for (int c = 0; c < easypap_mesh_desc.nb_cells; c++)
      border_info[c] |= binfo[c];
  }

  if (halo > 1)
    free (binfo);

  free (prefix);
  free (border_info);
  free (tmp_metap_of_cell);
}

static void mesh_mgpu_alloc_buffers (void)
{
  // Allocate GPU buffers
  for (int mp = 0; mp < easypap_mesh_desc.nb_metap; mp++) {

    gpu_data[mp].incoming_values =
        malloc (mesh_mgpu_info[mp].incoming_size * sizeof (float));

    gpu_data[mp].outgoing_values =
        malloc (mesh_mgpu_info[mp].outgoing_size * sizeof (float));

    gpu_data[mp].outgoing_index_buffer = clCreateBuffer (
        context, CL_MEM_READ_WRITE,
        mesh_mgpu_info[mp].outgoing_size * sizeof (unsigned), NULL, NULL);
    if (!gpu_data[mp].outgoing_index_buffer)
      exit_with_error ("Failed to allocate outgoing index buffer %d", mp);

    // Transfer outgoing indexes
    cl_int err = clEnqueueWriteBuffer (
        ocl_gpu[mp].q, gpu_data[mp].outgoing_index_buffer, CL_TRUE, 0,
        mesh_mgpu_info[mp].outgoing_size * sizeof (unsigned),
        gpu_data[mp].outgoing_index, 0, NULL, NULL);
    check (err, "Failed to write to outgoing index buffer");

    gpu_data[mp].outgoing_values_buffer = clCreateBuffer (
        context, CL_MEM_READ_WRITE,
        mesh_mgpu_info[mp].outgoing_size * sizeof (float), NULL, NULL);
    if (!gpu_data[mp].outgoing_values_buffer)
      exit_with_error ("Failed to allocate outgoing values buffer %d", mp);
  }

  // neighbors
  int *newpos = malloc (easypap_mesh_desc.nb_cells * sizeof (int));

  for (int mp = 0; mp < easypap_mesh_desc.nb_metap; mp++) {

    // reset newpos
    for (int c = 0; c < easypap_mesh_desc.nb_cells; c++)
      newpos[c] = -1;

    // shift main cells
    for (int c = 0; c < mesh_mgpu_info[mp].nb_cells; c++)
      newpos[mesh_mgpu_info[mp].part_offset + c] = c;

    // locate borders
    for (int n_mp = 0; n_mp < easypap_mesh_desc.nb_metap; n_mp++)
      for (int i = mesh_mgpu_info[n_mp].outgoing_offsets[mp];
           i < mesh_mgpu_info[n_mp].outgoing_offsets[mp + 1]; i++) {
        int src =
            gpu_data[n_mp].outgoing_index[i] + mesh_mgpu_info[n_mp].part_offset;
        int dst     = mesh_mgpu_info[mp].incoming_offsets[n_mp] + i;
        newpos[src] = dst;
#if 0
        PRINT_DEBUG ('m', "MP%d: newpos[%d] <- %d\n", mp, src, dst);
#endif
      }

    gpu_data[mp].neighbor_offset = mesh_mgpu_nb_threads (mp);
    const unsigned size = gpu_data[mp].neighbor_offset *
                          easypap_mesh_desc.max_neighbors * sizeof (int);
    gpu_data[mp].neighbor_soa = malloc (size);

    // Translate neighbors for legacy cells
    for (int c = mesh_mgpu_info[mp].part_offset;
         c < mesh_mgpu_info[mp].part_offset + mesh_mgpu_info[mp].nb_cells;
         c++) {
      int index = easypap_mesh_desc.index_first_neighbor[c];
      int n     = 0;
      while (index < easypap_mesh_desc.index_first_neighbor[c + 1]) {
        int np = newpos[easypap_mesh_desc.neighbors[index]];
        if (np == -1)
          exit_with_error ("unexpected new position (%d -> %d), needed for "
                           "%dth neighbor of %d",
                           easypap_mesh_desc.neighbors[index], np, n, c);
        gpu_data[mp].neighbor_soa[n * gpu_data[mp].neighbor_offset + c -
                                  mesh_mgpu_info[mp].part_offset] = np;
        index++;
        n++;
      }
      while (n < easypap_mesh_desc.max_neighbors) {
        gpu_data[mp].neighbor_soa[n * gpu_data[mp].neighbor_offset + c -
                                  mesh_mgpu_info[mp].part_offset] = -1;
        n++;
      }
#if 0
      PRINT_DEBUG (
          'm', "MP%d: cell %d neighbors: %d %d %d\n", mp,
          c - mesh_mgpu_info[mp].part_offset,
          gpu_data[mp].neighbor_soa[0 * gpu_data[mp].neighbor_offset + c -
                                    mesh_mgpu_info[mp].part_offset],
          gpu_data[mp].neighbor_soa[1 * gpu_data[mp].neighbor_offset + c -
                                    mesh_mgpu_info[mp].part_offset],
          gpu_data[mp].neighbor_soa[2 * gpu_data[mp].neighbor_offset + c -
                                    mesh_mgpu_info[mp].part_offset]);
#endif
    }

    int dest = mesh_mgpu_info[mp].nb_cells;
    for (int n_mp = 0; n_mp < easypap_mesh_desc.nb_metap; n_mp++)
      for (int i = mesh_mgpu_info[n_mp].outgoing_offsets[mp];
           i < mesh_mgpu_info[n_mp].outgoing_offsets[mp + 1]; i++) {
        int c =
            gpu_data[n_mp].outgoing_index[i] + mesh_mgpu_info[n_mp].part_offset;
        int index = easypap_mesh_desc.index_first_neighbor[c];
        int n     = 0;
        while (index < easypap_mesh_desc.index_first_neighbor[c + 1]) {
          int np = newpos[easypap_mesh_desc.neighbors[index]];
          gpu_data[mp].neighbor_soa[n * gpu_data[mp].neighbor_offset + dest] =
              np;
          index++;
          n++;
        }
        while (n < easypap_mesh_desc.max_neighbors) {
          gpu_data[mp].neighbor_soa[n * gpu_data[mp].neighbor_offset + dest] =
              -1;
          n++;
        }
#if 0
        PRINT_DEBUG (
            'm', "MP%d: cell %d neighbors: %d %d %d\n", mp, dest,
            gpu_data[mp].neighbor_soa[0 * gpu_data[mp].neighbor_offset + dest],
            gpu_data[mp].neighbor_soa[1 * gpu_data[mp].neighbor_offset + dest],
            gpu_data[mp].neighbor_soa[2 * gpu_data[mp].neighbor_offset + dest]);
#endif
        dest++;
      }

    gpu_data[mp].soa_buffer =
        clCreateBuffer (context, CL_MEM_READ_ONLY, size, NULL, NULL);
    if (!gpu_data[mp].soa_buffer)
      exit_with_error ("Failed to allocate SOA neighbor buffer\n");

    // Transfer neighbor SOA
    cl_int err = clEnqueueWriteBuffer (
        ocl_gpu[mp].q, gpu_data[mp].soa_buffer, CL_TRUE, 0, size,
        gpu_data[mp].neighbor_soa, 0, NULL, NULL);
    check (err, "Failed to write to SOA neighbor buffer");
  }

  free (newpos);

  for (int g = 0; g < easypap_mesh_desc.nb_metap; g++) {
    cl_int err;
    gpu_data[g].gather_kernel =
        clCreateKernel (program, "gather_outgoing_cells", &err);
    check (err, "Failed to create gather_outgoing_buffer kernel");
  }
}

void mesh_mgpu_config (unsigned halo_width)
{
  if (easypap_mesh_desc.nb_metap < 2)
    exit_with_error ("Mesh should at least be (meta)partitionned into 2 "
                     "domains (nb_metap == %d)",
                     easypap_mesh_desc.nb_metap);

  if (halo_width < 1)
    exit_with_error ("Halo width must be greater of equal to 1");

  halo = halo_width;

  PRINT_DEBUG ('m', "Mesh inter-domain halo width: %d\n", halo);

  if (picking_enabled) {
    mgpu_hud = ezv_hud_alloc (ctx[0]);
    ezv_hud_on (ctx[0], mgpu_hud);
  }
}

void mesh_mgpu_init (void)
{
  mesh_mgpu_build_info ();
  mesh_mgpu_alloc_buffers ();

  if (debug_enabled ('m'))
    mesh_mgpu_display_stats ();
}

void mesh_mgpu_exchg_borders (void)
{
  // gather outgoing cells
  size_t global[1], local[1] = {TILE};
  cl_int err;

  for (int g = 0; g < easypap_mesh_desc.nb_metap; g++) {

    global[0] = ROUND_TO_MULTIPLE (mesh_mgpu_info[g].outgoing_size, TILE);

    // Set gather kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (gpu_data[g].gather_kernel, 0, sizeof (cl_mem),
                           &ocl_gpu[g].curb);
    err |= clSetKernelArg (gpu_data[g].gather_kernel, 1, sizeof (cl_mem),
                           &gpu_data[g].outgoing_index_buffer);
    err |= clSetKernelArg (gpu_data[g].gather_kernel, 2, sizeof (cl_mem),
                           &gpu_data[g].outgoing_values_buffer);
    err |= clSetKernelArg (gpu_data[g].gather_kernel, 3, sizeof (unsigned),
                           &mesh_mgpu_info[g].outgoing_size);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (ocl_gpu[g].q, gpu_data[g].gather_kernel, 1,
                                  NULL, global, local, 0, NULL, NULL);
    check (err, "Failed to execute kernel");
  }

  // Retrieve values to RAM
  for (int g = 0; g < easypap_mesh_desc.nb_metap; g++) {
    err = clEnqueueReadBuffer (ocl_gpu[g].q, gpu_data[g].outgoing_values_buffer,
                               CL_TRUE, 0,
                               mesh_mgpu_info[g].outgoing_size * sizeof (float),
                               gpu_data[g].outgoing_values, 0, NULL, NULL);
    check (err, "Failed to read from outgoing values buffer");
  }

  for (int mp = 0; mp < easypap_mesh_desc.nb_metap; mp++) {
    int index = 0;
    for (int n_mp = 0; n_mp < easypap_mesh_desc.nb_metap; n_mp++)
      for (int i = mesh_mgpu_info[n_mp].outgoing_offsets[mp];
           i < mesh_mgpu_info[n_mp].outgoing_offsets[mp + 1]; i++)
        gpu_data[mp].incoming_values[index++] =
            gpu_data[n_mp].outgoing_values[i];
  }

  // Push borders to GPU
  for (int g = 0; g < easypap_mesh_desc.nb_metap; g++) {
    err =
        clEnqueueWriteBuffer (ocl_gpu[g].q, ocl_gpu[g].curb, CL_FALSE,
                              mesh_mgpu_info[g].nb_cells * sizeof (float),
                              mesh_mgpu_info[g].incoming_size * sizeof (float),
                              gpu_data[g].incoming_values, 0, NULL, NULL);
    check (err, "Failed to write borders to current buffer");
  }
}

void mesh_mgpu_send_initial_data (void)
{
  for (int g = 0; g < easypap_mesh_desc.nb_metap; g++) {
    cl_int err = clEnqueueWriteBuffer (
        ocl_gpu[g].q, ocl_gpu[g].curb, CL_TRUE,
        0, // offset is 0
        mesh_mgpu_info[g].nb_cells * sizeof (float),
        mesh_data + mesh_mgpu_info[g].part_offset, 0, NULL, NULL);
    check (err, "Failed to write to cur_data buffer");
  }

  mesh_mgpu_exchg_borders ();
}

void mesh_mgpu_refresh_img (void)
{
  for (int g = 0; g < easypap_mesh_desc.nb_metap; g++) {
    cl_int err = clEnqueueReadBuffer (
        ocl_gpu[g].q, ocl_gpu[g].curb, CL_TRUE, 0, // offset is zero
        mesh_mgpu_info[g].nb_cells * sizeof (float),
        mesh_data + mesh_mgpu_info[g].part_offset, 0, NULL, NULL);
    check (err, "Failed to read from cur_data buffer");
  }
}

void mesh_mgpu_debug (int cell)
{
  if (mgpu_hud != -1) {
    if (cell == -1)
      ezv_hud_off (ctx[0], mgpu_hud);
    else {
      ezv_hud_on (ctx[0], mgpu_hud);
      ezv_hud_set (ctx[0], mgpu_hud, "GPU: %d",
                   mesh3d_obj_get_metap_of_patch (
                       &easypap_mesh_desc, mesh3d_obj_get_patch_of_cell (
                                               &easypap_mesh_desc, cell)));
    }
  }
}

void mesh_mgpu_overlay (int cell)
{
  if (easypap_mesh_desc.nb_metap > 1) {
    int mpart = mesh3d_obj_get_metap_of_patch (
        &easypap_mesh_desc,
        mesh3d_obj_get_patch_of_cell (&easypap_mesh_desc, cell));
    int p1 = easypap_mesh_desc.metap_first_patch[mpart];
    int p2 = easypap_mesh_desc.metap_first_patch[mpart + 1];
    int c1 = easypap_mesh_desc.patch_first_cell[p1];
    int c2 = easypap_mesh_desc.patch_first_cell[p2];

    // highlight meta-partition
    ezv_set_cpu_color_1D (ctx[0], c1, c2 - c1,
                          ezv_rgba (0xC0, 0xC0, 0xC0, 0xC0));

    // highlight internal border
    for (int i = 0; i < mesh_mgpu_info[mpart].outgoing_size; i++) {
      int cell =
          gpu_data[mpart].outgoing_index[i] + mesh_mgpu_info[mpart].part_offset;
      ezv_set_cpu_color_1D (ctx[0], cell, 1, ezv_rgba (0xFF, 0xFF, 0x00, 0xC0));
    }

    // external border
    // for (int n_mp = 0; n_mp < mesh.nb_metap; n_mp++)
    //   for (int i = mesh_mgpu_info[n_mp].outgoing_offsets[mpart];
    //        i < mesh_mgpu_info[n_mp].outgoing_offsets[mpart + 1]; i++) {
    //     int cell =
    //         gpu[n_mp].outgoing_index[i] + mesh_mgpu_info[n_mp].part_offset;
    //     ezv_set_cpu_color_1D (ctx[0], cell, 1, 0xFFFF00F0);
    //   }

    // highlight cell
    ezv_set_cpu_color_1D (ctx[0], cell, 1, ezv_rgb (0xFF, 0x00, 0x00));
  }
}

#endif