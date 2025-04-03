#include "kernel/ocl/common.cl"

__kernel void life_ocl(__global unsigned *in, __global unsigned *out)
{
    // Define a tile in local memory with a 1-cell halo around it for neighbors
    __local unsigned tile[TILE_H + 2][TILE_W + 2];
    
    // Global and local coordinates
    unsigned x = get_global_id(0);
    unsigned y = get_global_id(1);
    unsigned xloc = get_local_id(0) + 1;  // +1 to account for halo
    unsigned yloc = get_local_id(1) + 1;
    unsigned local_size_x = get_local_size(0);
    unsigned local_size_y = get_local_size(1);
    
    // Global dimensions (assuming DIM is defined)
    unsigned width = DIM;
    unsigned height = DIM;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Load main tile area
    if (x < width && y < height) {
        tile[yloc][xloc] = in[y * width + x];
    }
    
    // Load halo regions - this requires cooperation between work items
    // Left halo
    if (xloc == 1) {
        unsigned left_x = (x == 0) ? width - 1 : x - 1;
        tile[yloc][0] = in[y * width + left_x];
    }
    
    // Right halo
    if (xloc == local_size_x) {
        unsigned right_x = (x == width - 1) ? 0 : x + 1;
        tile[yloc][local_size_x + 1] = in[y * width + right_x];
    }
    
    // Top halo
    if (yloc == 1) {
        unsigned top_y = (y == 0) ? height - 1 : y - 1;
        tile[0][xloc] = in[top_y * width + x];
    }
    
    // Bottom halo
    if (yloc == local_size_y) {
        unsigned bottom_y = (y == height - 1) ? 0 : y + 1;
        tile[local_size_y + 1][xloc] = in[bottom_y * width + x];
    }
    
    // Corner halos (need one work item to handle each)
    if (xloc == 1 && yloc == 1) {
        // Top-left
        unsigned tl_x = (x == 0) ? width - 1 : x - 1;
        unsigned tl_y = (y == 0) ? height - 1 : y - 1;
        tile[0][0] = in[tl_y * width + tl_x];
    }
    if (xloc == local_size_x && yloc == 1) {
        // Top-right
        unsigned tr_x = (x == width - 1) ? 0 : x + 1;
        unsigned tr_y = (y == 0) ? height - 1 : y - 1;
        tile[0][local_size_x + 1] = in[tr_y * width + tr_x];
    }
    if (xloc == 1 && yloc == local_size_y) {
        // Bottom-left
        unsigned bl_x = (x == 0) ? width - 1 : x - 1;
        unsigned bl_y = (y == height - 1) ? 0 : y + 1;
        tile[local_size_y + 1][0] = in[bl_y * width + bl_x];
    }
    if (xloc == local_size_x && yloc == local_size_y) {
        // Bottom-right
        unsigned br_x = (x == width - 1) ? 0 : x + 1;
        unsigned br_y = (y == height - 1) ? 0 : y + 1;
        tile[local_size_y + 1][local_size_x + 1] = in[br_y * width + br_x];
    }
    
    // Wait for all work items to finish loading data
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Only process cells within the global bounds
    if (x < width && y < height) {
        // Count live neighbors (8 surrounding cells)
        unsigned neighbors = 
            tile[yloc-1][xloc-1] + tile[yloc-1][xloc] + tile[yloc-1][xloc+1] +
            tile[yloc][xloc-1]   +                       tile[yloc][xloc+1] +
            tile[yloc+1][xloc-1] + tile[yloc+1][xloc] + tile[yloc+1][xloc+1];
        
        // Apply Game of Life rules
        unsigned current = tile[yloc][xloc];
        unsigned new_state = 0;
        
        if (current == 1) {
            // Cell is alive
            new_state = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            // Cell is dead
            new_state = (neighbors == 3) ? 1 : 0;
        }
        
        // Write result to global memory
        out[y * width + x] = new_state;

        
    }
}


// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a life-specific version (generic version is defined in common.cl)
__kernel void life_update_texture (__global unsigned *cur,
                                   __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);

  write_imagef (tex, (int2)(x, y),
                color_to_float4 (cur[y * DIM + x] * rgb (255, 255, 0)));
}
