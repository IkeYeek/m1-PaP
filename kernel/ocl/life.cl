__kernel void life_ocl (__global unsigned *in, __global unsigned *out){

    int x = get_global_id (0);
    int y = get_global_id (1);

    unsigned next = in [y * DIM + x];

    out [y * DIM + x] = next;
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a life-specific version (generic version is defined in common.cl)
__kernel void life_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
    int y = get_global_id (1);
    int x = get_global_id (0);

    write_imagef (tex, (int2)(x, y), color_to_float4 (cur [y * DIM + x] * rgb (255, 255, 0)));
}