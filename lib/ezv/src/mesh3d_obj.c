#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "error.h"
#include "mesh3d_obj.h"
#include "mesh3d_palette.h"
#include "tinyobj_loader_c.h"
#ifdef USE_SCOTCH
#include "scotch.h"
#endif

static unsigned vi  = 0;
static unsigned ti  = 0;
static unsigned nbt = 0;
static unsigned nbv = 0;

void mesh3d_obj_init (mesh3d_obj_t *mesh)
{
  mesh->mesh_type            = MESH3D_TYPE_SURFACE;
  mesh->vertices             = NULL;
  mesh->nb_vertices          = 0;
  mesh->triangles            = NULL;
  mesh->nb_triangles         = 0;
  mesh->cells                = NULL;
  mesh->triangle_info        = NULL;
  mesh->nb_cells             = 0;
  mesh->min_neighbors        = 0;
  mesh->max_neighbors        = 0;
  mesh->total_neighbors      = 0;
  mesh->neighbors            = NULL;
  mesh->index_first_neighbor = NULL;
  mesh->nb_patches           = 0;
  mesh->patch_first_cell     = NULL;

  vi  = 0;
  ti  = 0;
  nbv = 0;
  nbt = 0;
}

static void calculate_bounding_box (mesh3d_obj_t *mesh)
{
  bbox_t *bbox = &mesh->bbox;

  bbox->min[0] = bbox->max[0] = mesh->vertices[0];
  bbox->min[1] = bbox->max[1] = mesh->vertices[1];
  bbox->min[2] = bbox->max[2] = mesh->vertices[2];
  for (int v = 1; v < mesh->nb_vertices; v++)
    for (int c = 0; c < 3; c++) {
      if (mesh->vertices[3 * v + c] < bbox->min[c])
        bbox->min[c] = mesh->vertices[3 * v + c];
      if (mesh->vertices[3 * v + c] > bbox->max[c])
        bbox->max[c] = mesh->vertices[3 * v + c];
    }
  // Debug:
  // printf ("Min[%f,%f,%f] -> Max[%f,%f,%f]\n", bbox->min[0], bbox->min[1],
  // bbox->min[2], bbox->max[0], bbox->max[1], bbox->max[2]);
}

static void display_stats (mesh3d_obj_t *mesh)
{
  printf ("Mesh: %d cells, %d triangles, %d vertices, %d total neighbors\n",
          mesh->nb_cells, mesh->nb_triangles, mesh->nb_vertices,
          mesh->total_neighbors);
  if (mesh->nb_patches > 0)
    printf ("Mesh already partitionned into %d patches\n", mesh->nb_patches);
}

static int add_vertice (mesh3d_obj_t *mesh, float x, float y, float z)
{
  mesh->vertices[vi++] = x;
  mesh->vertices[vi++] = y;
  mesh->vertices[vi++] = z;

  return nbv++;
}

static int add_triangle (mesh3d_obj_t *mesh, unsigned v1, unsigned v2,
                         unsigned v3)
{
  mesh->triangles[ti++] = v1;
  mesh->triangles[ti++] = v2;
  mesh->triangles[ti++] = v3;

  return nbt++;
}

static void mesh3d_form_cells (mesh3d_obj_t *mesh, unsigned group_size)
{
  unsigned indt = 0;
  unsigned c;

  if (group_size < 1)
    exit_with_error (
        "mesh3d_form_cells: group_size (%d) must be greater than 1",
        group_size);

  if (mesh->nb_triangles % group_size != 0)
    exit_with_error ("mesh3d_form_cells: #triangles (%d) is not a multiple of "
                     "group_size (%d)",
                     mesh->nb_triangles, group_size);

  mesh->nb_cells = mesh->nb_triangles / group_size;
  // cells is an array of indexes. Cells[i] contains the index of its first
  // triangle. Triangles belonging to the same cell are stored contiguously. We
  // allocate one more index so that the triangle range is simply
  // cell[i]..cell[i+1]-1
  mesh->cells = malloc ((mesh->nb_cells + 1) * sizeof (unsigned));
  if (mesh->triangle_info == NULL) {
    printf ("Lazy alloc!\n");
    mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));
  }
  for (c = 0; c < mesh->nb_cells; c++) {
    mesh->cells[c] = indt;
    for (int g = 0; g < group_size; g++)
      mesh->triangle_info[indt++] |= (c << CELLNO_SHIFT);
  }
  // extra cell
  mesh->cells[c] = indt;
}

// /////////////// Cube

void mesh3d_obj_build_cube (mesh3d_obj_t *mesh, unsigned group_size)
{
  mesh->nb_vertices   = 8;
  mesh->vertices      = malloc (mesh->nb_vertices * 3 * sizeof (float));
  mesh->nb_triangles  = 12;
  mesh->triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));

  // We voluntarily use a scaling factor (*2) as well as a y-translation (+1) to
  // challenge the renderer viewport initialization ;)
  add_vertice (mesh, 2 * -0.5f, 1 + 2 * -0.5f, 2 * 0.5f); // 0
  add_vertice (mesh, 2 * 0.5f, 1 + 2 * -0.5f, 2 * 0.5f);  // 1
  add_vertice (mesh, 2 * -0.5f, 1 + 2 * 0.5f, 2 * 0.5f);  // 2
  add_vertice (mesh, 2 * 0.5f, 1 + 2 * 0.5f, 2 * 0.5f);   // 3

  add_vertice (mesh, 2 * -0.5f, 1 + 2 * -0.5f, 2 * -0.5f); // 4
  add_vertice (mesh, 2 * 0.5f, 1 + 2 * -0.5f, 2 * -0.5f);  // 5
  add_vertice (mesh, 2 * -0.5f, 1 + 2 * 0.5f, 2 * -0.5f);  // 6
  add_vertice (mesh, 2 * 0.5f, 1 + 2 * 0.5f, 2 * -0.5f);   // 7

  if (mesh->nb_vertices != nbv)
    exit_with_error ("mesh3d: the final number of vertices (%d) does not match "
                     "the initial one (%d)",
                     nbv, mesh->nb_vertices);

  add_triangle (mesh, 0, 2, 1); // front
  add_triangle (mesh, 1, 2, 3); //
  add_triangle (mesh, 4, 6, 0); // left
  add_triangle (mesh, 0, 6, 2); //
  add_triangle (mesh, 2, 6, 3); // top
  add_triangle (mesh, 3, 6, 7); //
  add_triangle (mesh, 4, 0, 5); // bottom
  add_triangle (mesh, 5, 0, 1); //
  add_triangle (mesh, 1, 3, 5); // right
  add_triangle (mesh, 5, 3, 7); //
  add_triangle (mesh, 5, 7, 4); // back
  add_triangle (mesh, 4, 7, 6); //

  if (group_size == 2) {
    for (int t = 0; t < mesh->nb_triangles; t++)
      if ((t & 1) == 0)
        mesh->triangle_info[t] |= EDGE1;
      else
        mesh->triangle_info[t] |= EDGE0;
  }

  if (mesh->nb_triangles != nbt)
    exit_with_error (
        "mesh3d: the final number of triangles (%d) does not match "
        "the initial one (%d)",
        nbt, mesh->nb_triangles);

  mesh3d_form_cells (mesh, group_size);

  // printf ("Mesh: %d vertices, %d cells, %d triangles\n", mesh->nb_vertices,
  //         mesh->nb_cells, mesh->nb_triangles);

  // Neighbors
  if (group_size == 2) {
    int index           = 0;
    mesh->min_neighbors = 4;
    mesh->max_neighbors = 4;
    mesh->neighbors =
        malloc (mesh->nb_cells * mesh->max_neighbors * sizeof (int));
    mesh->index_first_neighbor = malloc ((mesh->nb_cells + 1) * sizeof (int));
    // face 0 = front
    mesh->index_first_neighbor[0] = index;
    mesh->neighbors[index++]      = 1; // left
    mesh->neighbors[index++]      = 4; // right
    mesh->neighbors[index++]      = 2; // top
    mesh->neighbors[index++]      = 3; // bottom
    // face 1 = left
    mesh->index_first_neighbor[1] = index;
    mesh->neighbors[index++]      = 0; // front
    mesh->neighbors[index++]      = 5; // back
    mesh->neighbors[index++]      = 2; // top
    mesh->neighbors[index++]      = 3; // bottom
    // face = top
    mesh->index_first_neighbor[2] = index;
    mesh->neighbors[index++]      = 1; // left
    mesh->neighbors[index++]      = 4; // right
    mesh->neighbors[index++]      = 0; // front
    mesh->neighbors[index++]      = 5; // back
    // face 3 = bottom
    mesh->index_first_neighbor[3] = index;
    mesh->neighbors[index++]      = 1; // left
    mesh->neighbors[index++]      = 4; // right
    mesh->neighbors[index++]      = 0; // front
    mesh->neighbors[index++]      = 5; // back
    // face 4 = right
    mesh->index_first_neighbor[4] = index;
    mesh->neighbors[index++]      = 0; // front
    mesh->neighbors[index++]      = 5; // back
    mesh->neighbors[index++]      = 2; // top
    mesh->neighbors[index++]      = 3; // bottom
    // face 5 = back
    mesh->index_first_neighbor[5] = index;
    mesh->neighbors[index++]      = 1; // left
    mesh->neighbors[index++]      = 4; // right
    mesh->neighbors[index++]      = 2; // top
    mesh->neighbors[index++]      = 3; // bottom

    mesh->index_first_neighbor[6] = index;
    mesh->total_neighbors         = index;
  } else {
    // TODO
    exit_with_error ("Not yet implemented");
  }

  calculate_bounding_box (mesh);
}

static void build_cubus_volumus (mesh3d_obj_t *mesh, unsigned nbx, unsigned nby,
                                 unsigned nbz)
{
  const unsigned line    = nbx + 1;
  const unsigned surface = (nbx + 1) * (nby + 1);
  const unsigned volume  = (nbx + 1) * (nby + 1) * (nbz + 1);

  mesh->mesh_type     = MESH3D_TYPE_VOLUME;
  mesh->nb_vertices   = volume;
  mesh->vertices      = malloc (mesh->nb_vertices * 3 * sizeof (float));
  mesh->nb_triangles  = 12 * nbx * nby * nbz;
  mesh->triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));

  mesh->min_neighbors = 3;
  mesh->max_neighbors = 6;

  float i, j, k;
  float xinc, yinc, zinc;

  xinc = 1.0f / (float)nbx;
  yinc = 1.0f / (float)nby;
  zinc = 1.0f / (float)nbz;

  // First, generate all vertices
  for (int z = nbz; z >= 0; z--) {
    k = -0.5f + z * zinc;
    for (int y = 0; y <= nby; y++) {
      j = -0.5f + y * yinc;
      for (int x = 0; x <= nbx; x++) {
        i = -0.5f + x * xinc;
        add_vertice (mesh, i, j, k);
      }
    }
  }

  if (mesh->nb_vertices != nbv)
    exit_with_error ("mesh3d: the final number of vertices (%d) does not match "
                     "the initial one (%d)",
                     nbv, mesh->nb_vertices);

  float v[8];

  // Now create triangles
  for (int z = 0; z < nbz; z++) {
    for (int y = 0; y < nby; y++) {
      for (int x = 0; x < nbx; x++) {
        v[0] = z * surface + y * line + x;
        v[1] = z * surface + y * line + (x + 1);
        v[2] = z * surface + (y + 1) * line + x;
        v[3] = z * surface + (y + 1) * line + (x + 1);

        v[4] = (z + 1) * surface + y * line + x;
        v[5] = (z + 1) * surface + y * line + (x + 1);
        v[6] = (z + 1) * surface + (y + 1) * line + x;
        v[7] = (z + 1) * surface + (y + 1) * line + (x + 1);

        // Create triangles Clockwise
        if (z > 0) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[0], v[2], v[1]); // front
        add_triangle (mesh, v[1], v[2], v[3]);

        if (x > 0) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[4], v[6], v[0]); // left
        add_triangle (mesh, v[0], v[6], v[2]);

        if (y < nby - 1) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[2], v[6], v[3]); // top
        add_triangle (mesh, v[3], v[6], v[7]);

        if (y > 0) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[4], v[0], v[5]); // bottom
        add_triangle (mesh, v[5], v[0], v[1]);

        if (x < nbx - 1) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[1], v[3], v[5]); // right
        add_triangle (mesh, v[5], v[3], v[7]);

        if (z < nbz - 1) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[5], v[7], v[4]); // back
        add_triangle (mesh, v[4], v[7], v[6]);
      }
    }
  }

  if (mesh->nb_triangles != nbt)
    exit_with_error (
        "mesh3d: the final number of triangles (%d) does not match "
        "the initial one (%d)",
        nbt, mesh->nb_triangles);

  calculate_bounding_box (mesh);
}

// /////////////// Wall

static void build_wall (mesh3d_obj_t *mesh, unsigned nbx, unsigned nby)
{
  const unsigned line    = nbx + 1;
  const unsigned surface = (nbx + 1) * (nby + 1);

  mesh->mesh_type     = MESH3D_TYPE_SURFACE;
  mesh->nb_vertices   = surface;
  mesh->vertices      = malloc (mesh->nb_vertices * 3 * sizeof (float));
  mesh->nb_triangles  = 2 * nbx * nby;
  mesh->triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));

  mesh->min_neighbors = 1;
  mesh->max_neighbors = 3;

  float i, j;
  float xinc, yinc;

  xinc = 1.0f / (float)nbx;
  yinc = 1.0f / (float)nby;

  // First, generate all vertices
  for (int y = 0; y <= nby; y++) {
    j = -0.5f + y * yinc;
    for (int x = 0; x <= nbx; x++) {
      i = -0.5f + x * xinc;
      add_vertice (mesh, 0.0f, j, -i);
    }
  }

  if (mesh->nb_vertices != nbv)
    exit_with_error ("mesh3d: the final number of vertices (%d) does not match "
                     "the initial one (%d)",
                     nbv, mesh->nb_vertices);

  float v[4];

  // Now create triangles
  for (int y = 0; y < nby; y++) {
    for (int x = 0; x < nbx; x++) {
      v[0] = y * line + x;
      v[1] = y * line + (x + 1);
      v[2] = (y + 1) * line + x;
      v[3] = (y + 1) * line + (x + 1);

      // Create triangles Clockwise
      add_triangle (mesh, v[0], v[2], v[1]); // front
      add_triangle (mesh, v[1], v[2], v[3]);
    }
  }

  if (mesh->nb_triangles != nbt)
    exit_with_error (
        "mesh3d: the final number of triangles (%d) does not match "
        "the initial one (%d)",
        nbt, mesh->nb_triangles);

  calculate_bounding_box (mesh);
}

// /////////////// Torus

static const unsigned SEGMENTS  = 4 * 32;
static const unsigned SLICES    = 4 * 64;
static const float TORUS_RADIUS = 1.5f;
static const float RADIUS       = 0.5f;

static void build_torus (mesh3d_obj_t *mesh, unsigned nbx, unsigned nby)
{
  const unsigned circle    = nbx;
  const unsigned enveloppe = nbx * nby;

  mesh->mesh_type     = MESH3D_TYPE_SURFACE;
  mesh->nb_vertices   = enveloppe;
  mesh->vertices      = malloc (mesh->nb_vertices * 3 * sizeof (float));
  mesh->nb_triangles  = 2 * enveloppe;
  mesh->triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));

  mesh->min_neighbors = 3;
  mesh->max_neighbors = 3;

  float i, j, k;
  float lon, lat;
  float loninc, latinc;
  float d;

  loninc = 2.0f * M_PI / (float)nbx;
  latinc = 2.0f * M_PI / (float)nby;

  // First, generate all vertices
  for (int y = 0; y < nby; y++) {
    lat = y * latinc;
    for (int x = 0; x < nbx; x++) {
      lon = x * loninc;
      d   = TORUS_RADIUS + cos (lat) * RADIUS;
      i   = d * cos (lon);
      j   = sin (lat) * RADIUS;
      k   = d * -sin (lon);
      add_vertice (mesh, i, j, k);
      lat += latinc / 2.0;
    }
  }

  if (mesh->nb_vertices != nbv)
    exit_with_error ("mesh3d: the final number of vertices (%d) does not match "
                     "the initial one (%d)",
                     nbv, mesh->nb_vertices);

  float v[4];

  // Now create triangles
  for (int x = 0; x < nbx; x++) {
    for (int y = 0; y < nby; y++) {
      v[0] = y * circle + x;
      v[1] = y * circle + ((x + 1) % nbx);
      v[2] = ((y + 1) % nby) * circle + x;
      v[3] = ((y + 1) % nby) * circle + ((x + 1) % nbx);

      // Create triangles Clockwise
      add_triangle (mesh, v[0], v[2], v[1]); // front
      add_triangle (mesh, v[1], v[2], v[3]);
    }
  }

  if (mesh->nb_triangles != nbt)
    exit_with_error (
        "mesh3d: the final number of triangles (%d) does not match "
        "the initial one (%d)",
        nbt, mesh->nb_triangles);

  calculate_bounding_box (mesh);
}

static void build_torus_volumus (mesh3d_obj_t *mesh, unsigned nbx, unsigned nby,
                                 unsigned nbz)
{
  const unsigned circle    = nbx;
  const unsigned enveloppe = nbx * nby;
  const unsigned volume    = nbx * nby * nbz;

  mesh->mesh_type     = MESH3D_TYPE_VOLUME;
  mesh->nb_vertices   = enveloppe * (nbz + 1);
  mesh->vertices      = malloc (mesh->nb_vertices * 3 * sizeof (float));
  mesh->nb_triangles  = 12 * volume;
  mesh->triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));

  mesh->min_neighbors = 5;
  mesh->max_neighbors = 6;

  float i, j, k;
  float lon, lat, rad, base_radius = 9;
  float loninc, latinc, radinc;
  float d;

  loninc = 2.0f * M_PI / (float)nbx;
  latinc = 2.0f * M_PI / (float)nby;
  radinc = -8.0f / (float)(nbz);

  // First, generate all vertices
  for (int z = 0; z <= nbz; z++) { //  Warning : not a torus on z-dimension!
    rad = base_radius + z * (radinc);
    for (int y = 0; y < nby; y++) {
      lat = y * latinc;
      for (int x = 0; x < nbx; x++) {
        lon = x * loninc;
        d   = 12.0f + cos (lat) * rad;
        i   = d * cos (lon);
        j   = sin (lat) * rad;
        k   = d * -sin (lon);
        add_vertice (mesh, i, j, k);
      }
    }
  }

  if (mesh->nb_vertices != nbv)
    exit_with_error ("mesh3d: the final number of vertices (%d) does not match "
                     "the initial one (%d)",
                     nbv, mesh->nb_vertices);

  float v[8];

  // Now create triangles
  for (int z = 0; z < nbz; z++) {
    for (int y = 0; y < nby; y++) {
      for (int x = 0; x < nbx; x++) {
        v[0] = z * enveloppe + y * circle + x;
        v[1] = z * enveloppe + y * circle + ((x + 1) % nbx);
        v[2] = z * enveloppe + ((y + 1) % nby) * circle + x;
        v[3] = z * enveloppe + ((y + 1) % nby) * circle + ((x + 1) % nbx);

        v[4] = (z + 1) * enveloppe + y * circle + x;
        v[5] = (z + 1) * enveloppe + y * circle + ((x + 1) % nbx);
        v[6] = (z + 1) * enveloppe + ((y + 1) % nby) * circle + x;
        v[7] = (z + 1) * enveloppe + ((y + 1) % nby) * circle + ((x + 1) % nbx);

        // Create triangles Clockwise
        if (z > 0) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[0], v[2], v[1]); // front
        add_triangle (mesh, v[1], v[2], v[3]);

        mesh->triangle_info[nbt]     = 1;
        mesh->triangle_info[nbt + 1] = 1;
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[4], v[6], v[0]); // left
        add_triangle (mesh, v[0], v[6], v[2]);

        mesh->triangle_info[nbt]     = 1;
        mesh->triangle_info[nbt + 1] = 1;
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[2], v[6], v[3]); // top
        add_triangle (mesh, v[3], v[6], v[7]);

        mesh->triangle_info[nbt]     = 1;
        mesh->triangle_info[nbt + 1] = 1;
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[4], v[0], v[5]); // bottom
        add_triangle (mesh, v[5], v[0], v[1]);

        mesh->triangle_info[nbt]     = 1;
        mesh->triangle_info[nbt + 1] = 1;
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[1], v[3], v[5]); // right
        add_triangle (mesh, v[5], v[3], v[7]);

        if (z < nbz - 1) {
          mesh->triangle_info[nbt]     = 1;
          mesh->triangle_info[nbt + 1] = 1;
        }
        mesh->triangle_info[nbt] |= EDGE1;
        mesh->triangle_info[nbt + 1] |= EDGE0;
        add_triangle (mesh, v[5], v[7], v[4]); // back
        add_triangle (mesh, v[4], v[7], v[6]);
      }
    }
  }

  if (mesh->nb_triangles != nbt)
    exit_with_error (
        "mesh3d: the final number of triangles (%d) does not match "
        "the initial one (%d)",
        nbt, mesh->nb_triangles);

  calculate_bounding_box (mesh);
}

static void build_cylinder_volumus (mesh3d_obj_t *mesh, unsigned nbx,
                                    unsigned nby)
{
  const unsigned circle    = nbx;
  const unsigned enveloppe = nbx * (nby + 1);

  mesh->mesh_type     = MESH3D_TYPE_VOLUME;
  mesh->nb_vertices   = nbx * (nby + 1) * 2;
  mesh->vertices      = malloc (mesh->nb_vertices * 3 * sizeof (float));
  mesh->nb_triangles  = 12 * nbx * nby;
  mesh->triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));

  mesh->min_neighbors = 3;
  mesh->max_neighbors = 4;

  const float radius = 12.0;
  float i, j, k;
  float lon, h;
  float loninc, hinc, radinc;
  float d;

  loninc = 2.0f * M_PI / (float)nbx;
  hinc   = radius * sin (loninc) * 0.5;
  radinc = -hinc * 2;

  // First, generate all vertices
  for (int z = 0; z <= 1; z++) {
    d = radius + z * radinc;
    for (int y = 0; y <= nby; y++) {
      h = y * hinc;
      for (int x = 0; x < nbx; x++) {
        lon = x * loninc;
        i   = d * cos (lon);
        j   = h;
        k   = d * -sin (lon);
        add_vertice (mesh, i, j, k);
      }
    }
  }

  if (mesh->nb_vertices != nbv)
    exit_with_error ("mesh3d: the final number of vertices (%d) does not match "
                     "the initial one (%d)",
                     nbv, mesh->nb_vertices);

  float v[8];

  // Now create triangles
  for (int y = 0; y < nby; y++) {
    for (int x = 0; x < nbx; x++) {
      v[0] = y * circle + x;
      v[1] = y * circle + ((x + 1) % nbx);
      v[2] = (y + 1) * circle + x;
      v[3] = (y + 1) * circle + ((x + 1) % nbx);

      v[4] = enveloppe + y * circle + x;
      v[5] = enveloppe + y * circle + ((x + 1) % nbx);
      v[6] = enveloppe + (y + 1) * circle + x;
      v[7] = enveloppe + (y + 1) * circle + ((x + 1) % nbx);

      // Create triangles Clockwise
      mesh->triangle_info[nbt] |= EDGE1;
      mesh->triangle_info[nbt + 1] |= EDGE0;
      add_triangle (mesh, v[0], v[2], v[1]); // front
      add_triangle (mesh, v[1], v[2], v[3]);

      mesh->triangle_info[nbt]     = 1;
      mesh->triangle_info[nbt + 1] = 1;
      mesh->triangle_info[nbt] |= EDGE1;
      mesh->triangle_info[nbt + 1] |= EDGE0;
      add_triangle (mesh, v[4], v[6], v[0]); // left
      add_triangle (mesh, v[0], v[6], v[2]);

      if (y < nby - 1) {
        mesh->triangle_info[nbt]     = 1;
        mesh->triangle_info[nbt + 1] = 1;
      }
      mesh->triangle_info[nbt] |= EDGE1;
      mesh->triangle_info[nbt + 1] |= EDGE0;
      add_triangle (mesh, v[2], v[6], v[3]); // top
      add_triangle (mesh, v[3], v[6], v[7]);

      if (y > 0) {
        mesh->triangle_info[nbt]     = 1;
        mesh->triangle_info[nbt + 1] = 1;
      }
      mesh->triangle_info[nbt] |= EDGE1;
      mesh->triangle_info[nbt + 1] |= EDGE0;
      add_triangle (mesh, v[4], v[0], v[5]); // bottom
      add_triangle (mesh, v[5], v[0], v[1]);

      mesh->triangle_info[nbt]     = 1;
      mesh->triangle_info[nbt + 1] = 1;
      mesh->triangle_info[nbt] |= EDGE1;
      mesh->triangle_info[nbt + 1] |= EDGE0;
      add_triangle (mesh, v[1], v[3], v[5]); // right
      add_triangle (mesh, v[5], v[3], v[7]);

      mesh->triangle_info[nbt] |= EDGE1;
      mesh->triangle_info[nbt + 1] |= EDGE0;
      add_triangle (mesh, v[5], v[7], v[4]); // back
      add_triangle (mesh, v[4], v[7], v[6]);
    }
  }

  if (mesh->nb_triangles != nbt)
    exit_with_error (
        "mesh3d: the final number of triangles (%d) does not match "
        "the initial one (%d)",
        nbt, mesh->nb_triangles);

  calculate_bounding_box (mesh);
}

static unsigned morton2d (unsigned x, unsigned y)
{
  unsigned z = 0;

  for (int i = 0; i < 16; i++)
    z |= (x & 1U << i) << i | (y & 1U << i) << (i + 1);

  return z;
}

static void unmorton2d (unsigned z, unsigned *px, unsigned *py)
{
  unsigned x = 0, y = 0;
  unsigned m = 1;

  while (z) {
    x |= (z & 1) ? m : 0;
    y |= (z & 2) ? m : 0;
    z >>= 2;
    m <<= 1;
  }

  *px = x;
  *py = y;
}

static unsigned morton3d (unsigned x, unsigned y, unsigned z)
{
  unsigned n = 0;

  for (int i = 0; i < 10; i++)
    n |= (x & 1U << i) << (2 * i) | (y & 1U << i) << (2 * i + 1) |
         (z & 1U << i) << (2 * i + 2);

  return n;
}

static void unmorton3d (unsigned n, unsigned *px, unsigned *py, unsigned *pz)
{
  unsigned x = 0, y = 0, z = 0;
  unsigned mask = 1;

  while (n) {
    x |= (n & 1) ? mask : 0;
    y |= (n & 2) ? mask : 0;
    z |= (n & 4) ? mask : 0;

    n >>= 3;
    mask <<= 1;
  }

  *px = x;
  *py = y;
  *pz = z;
}

static unsigned neighbor_triangle (unsigned t, int num)
{
  int x, y;

  if ((t & 1) == 0) {
    if (num == 0)
      return t + 1; // up

    unsigned z = t >> 1;
    unmorton2d (z, (unsigned *)&x, (unsigned *)&y);

    if (num == 1) {
      x = (x - 1 + SLICES) % SLICES;
      z = morton2d (x, y);
      return (z << 1) + 1; // left
    }

    // num == 2
    y = (y - 1 + SEGMENTS) % SEGMENTS;
    z = morton2d (x, y);
    return (z << 1) + 1; // down
  } else {
    if (num == 2)
      return t - 1; // down

    unsigned z = t >> 1;
    unmorton2d (z, (unsigned *)&x, (unsigned *)&y);

    if (num == 1) {
      x = (x + 1 + SLICES) % SLICES;
      z = morton2d (x, y);
      return (z << 1); // right
    }

    // num == 0
    y = (y + 1 + SEGMENTS) % SEGMENTS;
    z = morton2d (x, y);
    return (z << 1); // up
  }
}

static unsigned neighbor_quad_geo (unsigned z, int dx, int dy)
{
  int x, y;

  unmorton2d (z, (unsigned *)&x, (unsigned *)&y);

  x = (x + dx + SLICES) % SLICES;
  y = (y + dy + SEGMENTS) % SEGMENTS;

  return morton2d (x, y);
}

static unsigned neighbor_quad (unsigned q, int num)
{
  switch (num) {
  case 0:
    return neighbor_quad_geo (q, 0, 1); // up
  case 1:
    return neighbor_quad_geo (q, 1, 0); // right
  case 2:
    return neighbor_quad_geo (q, 0, -1); // down
  case 3:
    return neighbor_quad_geo (q, -1, 0); // left
  default:
    exit_with_error ("quads have only 4 neighbors");
  }
}

static void sort_triangles_morton (mesh3d_obj_t *mesh, unsigned nbx,
                                   unsigned nby)
{
  unsigned t_src      = 0, t_dst;
  unsigned *triangles = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));

  for (unsigned x = 0; x < nbx; x++)
    for (unsigned y = 0; y < nby; y++) {
      unsigned z = morton2d (x, y);
      t_dst      = z * 6;
      for (int v = 0; v < 6; v++)
        triangles[t_dst + v] = mesh->triangles[t_src++];
    }
  free (mesh->triangles);
  mesh->triangles = triangles;
}

static void sort_cells_morton3d (mesh3d_obj_t *mesh, unsigned nbx, unsigned nby,
                                 unsigned nbz)
{
  unsigned v_src = 0, t_src = 0, t_dst;
  unsigned *triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  unsigned *triangle_info = malloc (mesh->nb_triangles * sizeof (unsigned));

  for (unsigned z = 0; z < nbz; z++)
    for (unsigned y = 0; y < nby; y++)
      for (unsigned x = 0; x < nbx; x++) {
        unsigned m = morton3d (x, y, z);
        if (m >= mesh->nb_triangles * 3 * 12)
          exit_with_error ("Morton2d number %d is too big! (#cells = %d)\n", m,
                           mesh->nb_cells);
        // copy vertices
        t_dst = m * 3 * 12;
        for (int v = 0; v < 3 * 12; v++)
          triangles[t_dst + v] = mesh->triangles[v_src++];
        // copy inner property
        t_dst = m * 12;
        for (int t = 0; t < 12; t++)
          triangle_info[t_dst + t] = mesh->triangle_info[t_src++];
      }
  free (mesh->triangles);
  mesh->triangles = triangles;

  free (mesh->triangle_info);
  mesh->triangle_info = triangle_info;
}

// We assume that cells are stored in a space-filling manner
static void build_neighbors_triangles (mesh3d_obj_t *mesh)
{
  int index = 0;

  mesh->min_neighbors = 3;
  mesh->max_neighbors = 3;
  mesh->neighbors =
      malloc (mesh->nb_cells * mesh->max_neighbors * sizeof (int));
  mesh->index_first_neighbor = malloc ((mesh->nb_cells + 1) * sizeof (int));

  for (int c = 0; c < mesh->nb_cells; c++) {
    mesh->index_first_neighbor[c] = index;
    for (int n = 0; n < 3; n++)
      mesh->neighbors[index++] = neighbor_triangle (c, n);
  }
  mesh->index_first_neighbor[mesh->nb_cells] = index;
  mesh->total_neighbors                      = index;
}

// We assume that cells are stored in a space-filling manner
static void build_neighbors_quads (mesh3d_obj_t *mesh)
{
  int index = 0;

  mesh->min_neighbors = 4;
  mesh->max_neighbors = 4; // each quad has 4 neighbors
  mesh->neighbors =
      malloc (mesh->nb_cells * mesh->max_neighbors * sizeof (int));
  mesh->index_first_neighbor = malloc ((mesh->nb_cells + 1) * sizeof (int));

  for (int c = 0; c < mesh->nb_cells; c++) {
    mesh->index_first_neighbor[c] = index;
    for (int n = 0; n < 4; n++)
      mesh->neighbors[index++] = neighbor_quad (c, n);
  }
  mesh->index_first_neighbor[mesh->nb_cells] = index;
  mesh->total_neighbors                      = index;
}

void mesh3d_obj_build_torus_surface (mesh3d_obj_t *mesh, unsigned group_size)
{
  build_torus (mesh, SLICES, SEGMENTS);

  sort_triangles_morton (mesh, SLICES, SEGMENTS);

  mesh3d_form_cells (mesh, group_size);

  if (group_size == 1) {
    build_neighbors_triangles (mesh);
  } else if (group_size == 2) {
    for (int t = 0; t < mesh->nb_triangles; t++) {
      if ((t & 1) == 0)
        mesh->triangle_info[t] |= EDGE1;
      else
        mesh->triangle_info[t] |= EDGE0;
    }
    build_neighbors_quads (mesh);
  }
}

// We assume that cells are stored in a space-filling manner
static void build_neighbors_3d (mesh3d_obj_t *mesh, unsigned size_x,
                                unsigned size_y, unsigned size_z, int toric)
{
  int index = 0;

  mesh->max_neighbors = 6; // each cube has 6 neighbors
  mesh->neighbors =
      malloc (mesh->nb_cells * mesh->max_neighbors * sizeof (int));
  mesh->index_first_neighbor = malloc ((mesh->nb_cells + 1) * sizeof (int));

  for (int c = 0; c < mesh->nb_cells; c++) {
    int x, y, z;
    unmorton3d (c, (unsigned *)&x, (unsigned *)&y, (unsigned *)&z);

    mesh->index_first_neighbor[c] = index;

    if (toric) {
      mesh->neighbors[index++] = morton3d ((x + size_x - 1) % size_x, y, z);
      mesh->neighbors[index++] = morton3d ((x + size_x + 1) % size_x, y, z);
      mesh->neighbors[index++] = morton3d (x, (y + size_y - 1) % size_y, z);
      mesh->neighbors[index++] = morton3d (x, (y + size_y + 1) % size_y, z);
    } else {
      if (x > 0) // left
        mesh->neighbors[index++] = morton3d (x - 1, y, z);
      if (x < size_x - 1) // right
        mesh->neighbors[index++] = morton3d (x + 1, y, z);
      if (y > 0) // down
        mesh->neighbors[index++] = morton3d (x, y - 1, z);
      if (y < size_y - 1) // up
        mesh->neighbors[index++] = morton3d (x, y + 1, z);
    }

    if (z > 0) // front
      mesh->neighbors[index++] = morton3d (x, y, z - 1);
    if (z < size_z - 1) // back
      mesh->neighbors[index++] = morton3d (x, y, z + 1);
  }
  mesh->index_first_neighbor[mesh->nb_cells] = index;
  mesh->total_neighbors                      = index;
}

void mesh3d_obj_build_cube_volume (mesh3d_obj_t *mesh, unsigned size)
{
  build_cubus_volumus (mesh, size, size, size);

  sort_cells_morton3d (mesh, size, size, size);

  mesh3d_form_cells (mesh, 12);

  build_neighbors_3d (mesh, size, size, size, 0);
}

void mesh3d_obj_build_wall (mesh3d_obj_t *mesh, unsigned size)
{
  build_wall (mesh, size, size);

  mesh3d_form_cells (mesh, 1);
}

void mesh3d_obj_build_torus_volume (mesh3d_obj_t *mesh, unsigned size_x,
                                    unsigned size_y, unsigned size_z)
{
  build_torus_volumus (mesh, size_x, size_y, size_z);

  sort_cells_morton3d (mesh, size_x, size_y, size_z);

  mesh3d_form_cells (mesh, 12);

  build_neighbors_3d (mesh, size_x, size_y, size_z, 1);
}

void mesh3d_obj_build_cylinder_volume (mesh3d_obj_t *mesh, unsigned size_x,
                                       unsigned size_y)
{
  build_cylinder_volumus (mesh, size_x, size_y);

  // sort_cells_morton2d (mesh, size_x, size_y);

  mesh3d_form_cells (mesh, 12);
}

// /////////////// Default mesh

void mesh3d_obj_build_default (mesh3d_obj_t *mesh)
{
  mesh3d_obj_build_torus_surface (mesh, 1);
}

// /////////////// OBJ file type

static char *obj_file = NULL;

static size_t file_size (const char *filename)
{
  struct stat sb;

  if (stat (filename, &sb) < 0)
    exit_with_error ("Cannot access \"%s\" file (%s)", filename,
                     strerror (errno));
  return sb.st_size;
}

static void get_file_data (void *ctx, const char *filename, const int is_mtl,
                           const char *obj_filename, char **data, size_t *len)
{
  (void)ctx;

  FILE *f;
  size_t s;
  size_t r;

  s = file_size (filename);
  if (obj_file != NULL) {
    fprintf (stderr, "Warning: OBJ file was not deleted\n");
    free (obj_file);
  }
  obj_file = malloc (s + 1);
  if (!obj_file)
    exit_with_error ("Malloc failed (%s)", strerror (errno));

  f = fopen (filename, "r");
  if (f == NULL)
    exit_with_error ("Cannot open \"%s\" file (%s)", filename,
                     strerror (errno));

  r = fread (obj_file, s, 1, f);
  if (r != 1)
    exit_with_error ("fread failed (%s)", strerror (errno));

  obj_file[s] = '\0';

  *data = obj_file;
  *len  = s;
}

static int triangles_share_edge (mesh3d_obj_t *mesh, int t1, int t2)
{
  int match = 0;

  for (int i1 = 0; i1 < 3; i1++) {
    int v = mesh->triangles[3 * t1 + i1];
    if (mesh->triangles[3 * t2 + 0] == v) { // match
      if (++match == 2)
        return 1; // triangles share a common edge (2 vertices)
    }
    if (mesh->triangles[3 * t2 + 1] == v) { // match
      if (++match == 2)
        return 1; // triangles share a common edge (2 vertices)
    }
    if (mesh->triangles[3 * t2 + 2] == v) { // match
      if (++match == 2)
        return 1; // triangles share a common edge (2 vertices)
    }

    if (match == 0 && i1 == 1)
      return 0; // no point in continuing the loop
  }
  return 0;
}

static unsigned triangles_common_edges (mesh3d_obj_t *mesh, int t1, int t2)
{
  unsigned vmatch = 0, edge_info = 0;

  for (unsigned i1 = 0; i1 < 3; i1++) {
    int v = mesh->triangles[3 * t1 + i1];
    if (mesh->triangles[3 * t2 + 0] == v || mesh->triangles[3 * t2 + 1] == v ||
        mesh->triangles[3 * t2 + 2] == v)
      vmatch |= (1 << i1);
  }

  if ((vmatch & 1) && (vmatch & 2))
    edge_info |= 1;
  if ((vmatch & 2) && (vmatch & 4))
    edge_info |= 2;
  if ((vmatch & 4) && (vmatch & 1))
    edge_info |= 4;

  return edge_info;
}

// No assumption about the order in which triangles are stored
// We use a N^2 algorithm :(
static void find_neighbors_triangles (mesh3d_obj_t *mesh)
{
  mesh->max_neighbors = 3;
  mesh->neighbors =
      malloc (mesh->nb_cells * mesh->max_neighbors * sizeof (int));
  mesh->index_first_neighbor = malloc ((mesh->nb_cells + 1) * sizeof (int));
  int *nb_neighbors          = calloc (mesh->nb_cells, sizeof (int));
  int *neighbors = malloc (mesh->nb_cells * mesh->max_neighbors * sizeof (int));

  for (int c1 = 0; c1 < mesh->nb_cells; c1++) {
    mesh->index_first_neighbor[c1] = 3 * c1;
    for (int c2 = 0; c2 < c1; c2++) {
      if (nb_neighbors[c2] == 3)
        continue;
      if (triangles_share_edge (mesh, c1, c2)) {
        neighbors[3 * c1 + nb_neighbors[c1]] = c2;
        neighbors[3 * c2 + nb_neighbors[c2]] = c1;
        ++nb_neighbors[c2];
        if (++nb_neighbors[c1] == 3)
          break;
      }
    }
  }

  // Reformat neighbors array
  int index = 0;
  for (int c = 0; c < mesh->nb_cells; c++) {
    mesh->index_first_neighbor[c] = index;
    for (int n = 0; n < nb_neighbors[c]; n++)
      mesh->neighbors[index++] = neighbors[3 * c + n];
  }

  mesh->index_first_neighbor[mesh->nb_cells] = index;
  mesh->total_neighbors                      = index;

  free (nb_neighbors);
  free (neighbors);
}

// #define REMOVE_VERTEX_DUPLICATES 1

#ifdef REMOVE_VERTEX_DUPLICATES
static int copy_without_duplicates (mesh3d_obj_t *mesh, float *src, int *vindex)
{
  int index = 0;
  for (int v = 0; v < mesh->nb_vertices; v++) {
    float c0 = src[3 * v + 0];
    float c1 = src[3 * v + 1];
    float c2 = src[3 * v + 2];
    int o;

    for (o = 0; o < index; o++)
      if (mesh->vertices[3 * o + 0] == c0 && mesh->vertices[3 * o + 1] == c1 &&
          mesh->vertices[3 * o + 2] == c2)
        // we found a duplicate
        break;

    vindex[v] = o;

    if (o == index) {
      // we didn't find any duplicate
      mesh->vertices[3 * index + 0] = c0;
      mesh->vertices[3 * index + 1] = c1;
      mesh->vertices[3 * index + 2] = c2;
      index++;
    }
  }
  return index;
}
#endif

static void load_obj_file (const char *filename, mesh3d_obj_t *mesh)
{
  // Tiny lib
  tinyobj_attrib_t attrib;
  tinyobj_shape_t *shapes = NULL;
  size_t num_shapes;
  tinyobj_material_t *materials = NULL;
  size_t num_materials;
  size_t face_offset = 0;
  int nbc            = 0;
  int index          = 0;

  int r = tinyobj_parse_obj (&attrib, &shapes, &num_shapes, &materials,
                             &num_materials, filename, get_file_data, NULL, 0);
  if (r != TINYOBJ_SUCCESS)
    exit_with_error ("tinyobj_parse_obj (%s) failure", filename);

  if (attrib.num_face_num_verts == 0)
    exit_with_error ("OBJ file %s seems empty", filename);

  mesh->mesh_type   = MESH3D_TYPE_SURFACE;
  mesh->nb_vertices = attrib.num_vertices;
  mesh->vertices    = malloc (mesh->nb_vertices * 3 * sizeof (float));
#ifdef REMOVE_VERTEX_DUPLICATES
  int *vindex = malloc (mesh->nb_vertices * sizeof (int));
#endif
  mesh->nb_triangles  = attrib.num_triangles;
  mesh->nb_cells      = attrib.num_face_num_verts;
  mesh->triangles     = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  mesh->triangle_info = calloc (mesh->nb_triangles, sizeof (unsigned));

  // cells is an array of indexes. Cells[i] contains the index of its first
  // triangle. Triangles belonging to the same cell are stored contiguously. We
  // allocate one more index so that the triangle range is simply
  // cell[i]..cell[i+1]-1
  mesh->cells = malloc ((mesh->nb_cells + 1) * sizeof (unsigned));

#ifdef REMOVE_VERTEX_DUPLICATES
  int d = copy_without_duplicates (mesh, attrib.vertices, vindex);
  if (d < mesh->nb_vertices) {
    printf ("Removed %d duplicates\n", mesh->nb_vertices - d);
    mesh->nb_vertices = d;
  } else
    printf ("Did not find any duplicate\n");
#else
#if 1
  // copy vertices coordinates as they are
  memcpy (mesh->vertices, attrib.vertices,
          mesh->nb_vertices * 3 * sizeof (float));
#else
  // Apply rotation on-the-fly
  for (int v = 0; v < mesh->nb_vertices; v++) {
    mesh->vertices[3 * v + 0] = -attrib.vertices[3 * v + 0];
    mesh->vertices[3 * v + 1] = attrib.vertices[3 * v + 1];
    mesh->vertices[3 * v + 2] = -attrib.vertices[3 * v + 2];
  }
#endif
#endif

  // Generate triangles and form cells from faces
  for (int i = 0; i < attrib.num_face_num_verts; i++) {
    if (attrib.face_num_verts[i] == 3) {
      int idx[3];

      for (int k = 0; k < 3; k++)
#ifdef REMOVE_VERTEX_DUPLICATES
        idx[k] = vindex[attrib.faces[face_offset + k].v_idx];
#else
        idx[k] = attrib.faces[face_offset + k].v_idx;
#endif

      mesh->cells[nbc++] = nbt;

      mesh->triangle_info[nbt] |= (nbc - 1) << CELLNO_SHIFT;
      add_triangle (mesh, idx[0], idx[1], idx[2]);

    } else if (attrib.face_num_verts[i] == 4) {
      int idx[4];

      for (int k = 0; k < 4; k++)
#ifdef REMOVE_VERTEX_DUPLICATES
        idx[k] = vindex[attrib.faces[face_offset + k].v_idx];
#else
        idx[k] = attrib.faces[face_offset + k].v_idx;
#endif

      mesh->cells[nbc++] = nbt;

      mesh->triangle_info[nbt] |= (EDGE1 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[0], idx[1], idx[3]);
      mesh->triangle_info[nbt] |= (EDGE0 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[3], idx[1], idx[2]);

    } else if (attrib.face_num_verts[i] == 8) {
      int idx[8];

      mesh->mesh_type = MESH3D_TYPE_VOLUME;

      for (int k = 0; k < 8; k++)
#ifdef REMOVE_VERTEX_DUPLICATES
        idx[k] = vindex[attrib.faces[face_offset + k].v_idx];
#else
        idx[k] = attrib.faces[face_offset + k].v_idx;
#endif

      mesh->cells[nbc++] = nbt;

      // front
      mesh->triangle_info[nbt] |= (EDGE1 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[0], idx[2], idx[1]);
      mesh->triangle_info[nbt] |= (EDGE0 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[1], idx[2], idx[3]);

      // left
      mesh->triangle_info[nbt] |= (EDGE1 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[4], idx[6], idx[0]);
      mesh->triangle_info[nbt] |= (EDGE0 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[0], idx[6], idx[2]);

      // top
      mesh->triangle_info[nbt] |= (EDGE1 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[2], idx[6], idx[3]);
      mesh->triangle_info[nbt] |= (EDGE0 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[3], idx[6], idx[7]);

      // bottom
      mesh->triangle_info[nbt] |= (EDGE1 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[4], idx[0], idx[5]);
      mesh->triangle_info[nbt] |= (EDGE0 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[5], idx[0], idx[1]);

      // right
      mesh->triangle_info[nbt] |= (EDGE1 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[1], idx[3], idx[5]);
      mesh->triangle_info[nbt] |= (EDGE0 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[5], idx[3], idx[7]);

      // back
      mesh->triangle_info[nbt] |= (EDGE1 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[5], idx[7], idx[4]);
      mesh->triangle_info[nbt] |= (EDGE0 | ((nbc - 1) << CELLNO_SHIFT));
      add_triangle (mesh, idx[4], idx[7], idx[6]);

    } else
      exit_with_error ("Unsupported face (#vertices = %d)",
                       attrib.face_num_verts[i]);

    face_offset += (size_t)attrib.face_num_verts[i];
  }

  // extra cell
  mesh->cells[nbc] = nbt;

#ifdef REMOVE_VERTEX_DUPLICATES
  free (vindex);
#endif

  if (nbt != attrib.num_triangles)
    exit_with_error (
        "mesh3d: the final number of triangles (%d) does not match "
        "the expected one (%d)",
        nbt, attrib.num_triangles);

  if (mesh->nb_cells != nbc)
    exit_with_error ("mesh3d: the final number of cells (%d) does not match "
                     "the initial one (%d)",
                     nbc, mesh->nb_cells);

  // Neighbors
  if (attrib.num_neighbors > 0) {
    if (attrib.num_neighbors != mesh->nb_cells)
      exit_with_error ("number of neighbor lines (%d) "
                       "different from #cells (%d)",
                       attrib.num_neighbors, mesh->nb_cells);

    mesh->max_neighbors = 0;
    mesh->min_neighbors = 6;

    mesh->neighbors = malloc (attrib.num_total_neighbors * sizeof (int));
    mesh->index_first_neighbor = malloc ((mesh->nb_cells + 1) * sizeof (int));

    for (int c = 0; c < mesh->nb_cells; c++) {
      if (attrib.connectivity_size[c] > mesh->max_neighbors)
        mesh->max_neighbors = attrib.connectivity_size[c];
      if (attrib.connectivity_size[c] < mesh->min_neighbors)
        mesh->min_neighbors = attrib.connectivity_size[c];

      mesh->index_first_neighbor[c] = index;

      for (int k = 0; k < attrib.connectivity_size[c]; k++)
        mesh->neighbors[index++] = attrib.connectivity[4 * c + k];
    }
    mesh->index_first_neighbor[mesh->nb_cells] = index;
    mesh->total_neighbors                      = index;
    if (mesh->total_neighbors != attrib.num_total_neighbors)
      exit_with_error ("number of neighbors was incorrectly calculated");
  }

  // Partitions
  if (attrib.num_patches != 0) {
    int indc               = 0;
    mesh->nb_patches       = attrib.num_patches;
    mesh->patch_first_cell = malloc ((mesh->nb_patches + 1) * sizeof (int));
    for (int p = 0; p < mesh->nb_patches; p++) {
      mesh->patch_first_cell[p] = indc;
      indc += attrib.patch[p];
    }
    mesh->patch_first_cell[mesh->nb_patches] = indc;

    if (indc != mesh->nb_cells)
      exit_with_error (
          "Sum of all partitions' sizes (%d) should be equal to #cells (%d)",
          indc, mesh->nb_cells);
  }

  calculate_bounding_box (mesh);

  // find_neighbors_triangles (mesh);

  tinyobj_attrib_free (&attrib);
  tinyobj_shapes_free (shapes, num_shapes);
  tinyobj_materials_free (materials, num_materials);

  if (obj_file != NULL) {
    free (obj_file);
    obj_file = NULL;
  }
}

// /////////////// load/store general functions

void mesh3d_obj_load (const char *filename, mesh3d_obj_t *mesh)
{
  if (!strcmp (filename, "1-torus.cgns"))
    mesh3d_obj_build_torus_surface (mesh, 1);
  else if (!strcmp (filename, "2-torus.cgns"))
    mesh3d_obj_build_torus_surface (mesh, 2);
  else if (!strcmp (filename, "3-torus.cgns"))
    mesh3d_obj_build_torus_volume (mesh, 64, 32, 32);
  else if (!strcmp (filename, "cyl.cgns"))
    mesh3d_obj_build_cylinder_volume (mesh, 400, 200);
  else if (strstr (filename, ".obj") != NULL)
    load_obj_file (filename, mesh);
  else
    exit_with_error ("mesh3d_obj_load can only load OBJ files");

  display_stats (mesh);
}

void mesh3d_obj_store (const char *filename, mesh3d_obj_t *mesh,
                       int with_patches)
{
  FILE *f = NULL;

  f = fopen (filename, "w");
  if (f == NULL)
    exit_with_error ("Cannot open \"%s\" file (%s)", filename,
                     strerror (errno));

  // Vertices
  for (int v = 0; v < mesh->nb_vertices; v++)
    fprintf (f, "v %f %f %f\n", mesh->vertices[3 * v + 0],
             mesh->vertices[3 * v + 1], mesh->vertices[3 * v + 2]);

  // Faces
  for (int c = 0; c < mesh->nb_cells; c++) {
    int nb_tri = mesh->cells[c + 1] - mesh->cells[c];
    if (nb_tri == 1) {
      fprintf (f, "f %d %d %d\n", mesh->triangles[3 * mesh->cells[c] + 0] + 1,
               mesh->triangles[3 * mesh->cells[c] + 1] + 1,
               mesh->triangles[3 * mesh->cells[c] + 2] + 1);

    } else if (nb_tri == 2) {
      // We keep t0[0 1] and t1[2 0]
      fprintf (f, "f %d %d %d %d\n",
               mesh->triangles[3 * (mesh->cells[c] + 0) + 0] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 0) + 1] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 1) + 2] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 1) + 0] + 1);

    } else if (nb_tri == 12) {
      // We keep front0[0 2 1], front1[2], back0[2 0] & back1[2 1]
      fprintf (f, "f %d %d %d %d %d %d %d %d\n",
               mesh->triangles[3 * (mesh->cells[c] + 0) + 0] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 0) + 2] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 0) + 1] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 1) + 2] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 10) + 2] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 10) + 0] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 11) + 2] + 1,
               mesh->triangles[3 * (mesh->cells[c] + 11) + 1] + 1);

    } else
      exit_with_error ("not yet implemented");
  }

  if (mesh->neighbors == NULL)
    find_neighbors_triangles (mesh);

  // Neighbors
  for (int c = 0; c < mesh->nb_cells; c++) {
    fprintf (f, "n");
    for (int n = mesh->index_first_neighbor[c];
         n < mesh->index_first_neighbor[c + 1]; n++)
      fprintf (f, " %d", mesh->neighbors[n] + 1);
    fprintf (f, "\n");
  }

  // Patches
  if (with_patches) {
    for (int p = 0; p < mesh->nb_patches; p++)
      fprintf (f, "p %d\n",
               mesh->patch_first_cell[p + 1] - mesh->patch_first_cell[p]);
  }
}

#define TIMESPEC2USEC(t)                                                       \
  ((uint64_t)(t).tv_sec * 1000000ULL + (t).tv_nsec / 1000)

static void reorder_cells (mesh3d_obj_t *mesh, int newpos[])
{
  struct timespec t1, t2;

  clock_gettime (CLOCK_MONOTONIC, &t1);
  int *oldpos = (int *)calloc (mesh->nb_cells, sizeof (int));

  for (int c = 0; c < mesh->nb_cells; c++)
    oldpos[newpos[c]] = c;

  unsigned *tmp_triangles = malloc (mesh->nb_triangles * 3 * sizeof (unsigned));
  unsigned *tmp_triangle_info = malloc (mesh->nb_triangles * sizeof (unsigned));
  unsigned *tmp_cells     = malloc ((mesh->nb_cells + 1) * sizeof (unsigned));
  unsigned *tmp_neighbors = malloc (mesh->total_neighbors * sizeof (int));
  unsigned *tmp_index_first_neightbor =
      malloc ((mesh->nb_cells + 1) * sizeof (int));
  unsigned ind_tri = 0;
  unsigned ind_nei = 0;

  for (int c = 0; c < mesh->nb_cells; c++) {
    unsigned op = oldpos[c];

    // move triangles
    tmp_cells[c] = ind_tri;
    for (int it = mesh->cells[op]; it < mesh->cells[op + 1]; it++) {
      tmp_triangles[3 * ind_tri + 0] = mesh->triangles[3 * it + 0];
      tmp_triangles[3 * ind_tri + 1] = mesh->triangles[3 * it + 1];
      tmp_triangles[3 * ind_tri + 2] = mesh->triangles[3 * it + 2];
      tmp_triangle_info[ind_tri] =
          (mesh->triangle_info[it] & ((1 << CELLNO_SHIFT) - 1)) |
          (c << CELLNO_SHIFT);
      ind_tri++;
    }

    // move neighbors
    // Warning: neighbors will move as well!
    tmp_index_first_neightbor[c] = ind_nei;
    for (int in = mesh->index_first_neighbor[op];
         in < mesh->index_first_neighbor[op + 1]; in++) {
      tmp_neighbors[ind_nei] = newpos[mesh->neighbors[in]];
      ind_nei++;
    }
  }
  tmp_cells[mesh->nb_cells]                 = ind_tri;
  tmp_index_first_neightbor[mesh->nb_cells] = ind_nei;

  if (ind_tri != mesh->nb_triangles)
    exit_with_error (
        "The total number of triangles is inconsistent (%d, should be %d)",
        ind_tri, mesh->nb_triangles);

  if (ind_nei != mesh->total_neighbors)
    exit_with_error (
        "The total number of neighbors is inconsistent (%d, should be %d)",
        ind_nei, mesh->total_neighbors);

  free (oldpos);

  free (mesh->cells);
  mesh->cells = tmp_cells;

  free (mesh->triangles);
  mesh->triangles = tmp_triangles;

  free (mesh->triangle_info);
  mesh->triangle_info = tmp_triangle_info;

  free (mesh->neighbors);
  mesh->neighbors = tmp_neighbors;

  free (mesh->index_first_neighbor);
  mesh->index_first_neighbor = tmp_index_first_neightbor;

  clock_gettime (CLOCK_MONOTONIC, &t2);

  printf ("Mesh cells reordered (%llu usec)\n",
          TIMESPEC2USEC (t2) - TIMESPEC2USEC (t1));
}

static void do_partition (mesh3d_obj_t *mesh, unsigned nbpart,
                          int use_partitionner)
{
  if (nbpart == 0)
    return;

  if (mesh->total_neighbors == 0)
    return; // Cannot partition when no information is given about neighbors

  if (mesh->nb_patches > 0) {
    printf ("Overriding existing partitionning…\n");
    if (mesh->patch_first_cell != NULL) {
      free (mesh->patch_first_cell);
      mesh->patch_first_cell = NULL;
    }
  }

  mesh->nb_patches       = nbpart;
  mesh->patch_first_cell = malloc ((nbpart + 1) * sizeof (int));

#ifdef USE_SCOTCH
  if (use_partitionner) {
    SCOTCH_Strat strategy;
    SCOTCH_Graph graph;

    SCOTCH_stratInit (&strategy); // Default strategy
    SCOTCH_graphInit (&graph);

    int r =
        SCOTCH_graphBuild (&graph,
                           0,                                        // baseval
                           mesh->nb_cells,                           // vertnbr
                           (SCOTCH_Num *)mesh->index_first_neighbor, // verttab
                           NULL,                                     // vendtab
                           NULL,                                     // velotab
                           NULL,                                     // vlbltab
                           mesh->total_neighbors,                    // edgenbr
                           (SCOTCH_Num *)mesh->neighbors,            // edgetab
                           NULL);                                    // edlotab
    if (r != 0)
      exit_with_error ("SCOTCH_graphBuild");

    int *parttab  = (int *)calloc (mesh->nb_cells, sizeof (int));
    int *newpos   = (int *)calloc (mesh->nb_cells, sizeof (int));
    int *partsize = calloc (nbpart, sizeof (int));
    int *prefix   = calloc (nbpart + 1, sizeof (int));

    struct timespec t1, t2;

    clock_gettime (CLOCK_MONOTONIC, &t1);

    r = SCOTCH_graphPart (&graph, nbpart, &strategy, parttab);
    if (r != 0)
      exit_with_error ("SCOTCH_graphPart");

    // free Scotch data structures
    SCOTCH_graphExit (&graph);
    SCOTCH_stratExit (&strategy);

    for (int c = 0; c < mesh->nb_cells; c++)
      partsize[parttab[c]]++;

    for (int p = 1; p < nbpart + 1; p++)
      prefix[p] = prefix[p - 1] + partsize[p - 1];

    // patch_first_cell is a prefix sum of partsize
    memcpy (mesh->patch_first_cell, prefix, (nbpart + 1) * sizeof (int));

    // calculate new indexes of cells
    for (int c = 0; c < mesh->nb_cells; c++)
      newpos[c] = prefix[parttab[c]]++;

    // We can get rid of some arrays at this point
    free (parttab);
    parttab = NULL;
    free (prefix);
    prefix = NULL;
    free (partsize);
    partsize = NULL;

    clock_gettime (CLOCK_MONOTONIC, &t2);

    printf ("Mesh partitionned into %d patches using Scotch (%llu usec)\n",
            nbpart, TIMESPEC2USEC (t2) - TIMESPEC2USEC (t1));

    // relayout everything :(
    reorder_cells (mesh, newpos);

    free (newpos);
  } else
#endif
  {
    if (use_partitionner)
      printf ("Warning: Falling back to a simple contiguous chunks "
              "distribution (USE_SCOTCH is not enabled)");

    // Build a straighforward array of patches (without changing cells order)
    const int chunk = mesh->nb_cells / nbpart;
    const int rem   = mesh->nb_cells % nbpart;

    int index = 0;
    for (int p = 0; p < nbpart; p++) {
      mesh->patch_first_cell[p] = index;
      index += chunk + (p < rem ? 1 : 0);
    }
    mesh->patch_first_cell[nbpart] = index;

    printf ("Mesh partitionned into %d chunks of contiguous cells\n", nbpart);
  }
}

void mesh3d_obj_partition (mesh3d_obj_t *mesh, unsigned nbpart, int flag)
{
  // Do actual partition
  do_partition (mesh, nbpart, flag & MESH3D_PART_USE_SCOTCH);

  if (flag & MESH3D_PART_SHOW_FRONTIERS) {
    // (re)compute parttab
    int *parttab = (int *)calloc (mesh->nb_cells, sizeof (int));
    for (int p = 0; p < mesh->nb_patches; p++)
      for (int c = mesh->patch_first_cell[p]; c < mesh->patch_first_cell[p + 1];
           c++)
        parttab[c] = p;

    // Find external frontiers of partitions
    for (int c = 0; c < mesh->nb_cells; c++)
      for (int t1 = mesh->cells[c]; t1 < mesh->cells[c + 1]; t1++) {
        int edge_info = 0;
        for (int n = mesh->index_first_neighbor[c];
             n < mesh->index_first_neighbor[c + 1]; n++)
          if (parttab[c] == parttab[mesh->neighbors[n]])
            for (int t2 = mesh->cells[mesh->neighbors[n]];
                 t2 < mesh->cells[mesh->neighbors[n] + 1]; t2++)
              edge_info |= triangles_common_edges (mesh, t1, t2);
        // edge_info indicates shared edges between cells of the same partition
        // so to hightlight frontiers, we need to invert edge_info bits
        mesh->triangle_info[t1] |= ((edge_info ^ 7U) << FRONTIER_SHIFT);
      }

    free (parttab);
  }
}

int mesh3d_obj_get_patch_of_cell (mesh3d_obj_t *mesh, unsigned cell)
{
  unsigned first = 0;
  unsigned last  = mesh->nb_patches - 1;

  while (first < last) {
    unsigned mid = (first + last) / 2;
    if (cell >= mesh->patch_first_cell[mid + 1])
      first = mid + 1;
    else
      last = mid;
  }

  if (cell >= mesh->patch_first_cell[first] &&
      cell < mesh->patch_first_cell[first + 1])
    return first;
  else
    return -1;
}

void mesh3d_obj_get_bbox_of_cell (mesh3d_obj_t *mesh, unsigned cell,
                                  bbox_t *box)
{
  int n = 0;

  if (cell >= mesh->nb_cells)
    exit_with_error ("Cell num (%d) exceeds total number of cells (%d)", cell,
                     mesh->nb_cells);

  for (int t = mesh->cells[cell]; t < mesh->cells[cell + 1]; t++) {
    for (int v = 0; v < 3; v++) {
      int vertex = mesh->triangles[3 * t + v];
      float coord[3];
      for (int c = 0; c < 3; c++)
        coord[c] = mesh->vertices[3 * vertex + c];
      if (!n) {
        for (int c = 0; c < 3; c++)
          box->min[c] = box->max[c] = coord[c];
        n = 1;
      } else {
        for (int c = 0; c < 3; c++) {
          box->min[c] = fmin (box->min[c], coord[c]);
          box->max[c] = fmax (box->max[c], coord[c]);
        }
      }
    }
  }
}

void mesh3d_obj_get_barycenter (mesh3d_obj_t *mesh, unsigned cell, float *bx,
                                float *by, float *bz)
{
  float n    = 0.0;
  float p[3] = {0.0, 0.0, 0.0};

  if (cell >= mesh->nb_cells)
    exit_with_error ("Cell num (%d) exceeds total number of cells (%d)", cell,
                     mesh->nb_cells);

  for (int t = mesh->cells[cell]; t < mesh->cells[cell + 1]; t++) {
    for (int v = 0; v < 3; v++) {
      int vertex = mesh->triangles[3 * t + v];
      for (int c = 0; c < 3; c++)
        p[c] += mesh->vertices[3 * vertex + c];
      n = n + 1.0;
    }
  }

  if (bx)
    *bx = p[0] / n;
  if (by)
    *by = p[1] / n;
  if (bz)
    *bz = p[2] / n;
}
