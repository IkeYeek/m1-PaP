#version 400 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

layout (std140) uniform Clipping
{
    float clipping_zcoord;
    float clipping_zproj;
    int clipping_active;
    float width;
    float height;
};

void main()
{    
    if (clipping_active > 0) {
        float s = 0.0;

        // First check if triangle should be discarded
        for (int i = 0; i < gl_in.length(); i++)
            s += step (clipping_zproj, gl_in[i].gl_Position.w);

        if (s == 0.0) // All points are in front of zplane
            return;
    }

    // vertex passthrough
    for (int i = 0; i < gl_in.length(); i++) {
        gl_Position = gl_in[i].gl_Position;
        gl_PrimitiveID = gl_PrimitiveIDIn;
        EmitVertex();
    }
}
