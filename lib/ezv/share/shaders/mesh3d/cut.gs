#version 400 core

// Per-triangle information (triangle_info field):
// bit  [0] = isInner
// bits [1..3] = <edge0, edge1, edge2>
// bits [4..6] = <frontier0, frontier1, frontier2>
// bits [7..31] = cell_no (limited to 2^25 = 32M cells)
#define ISINNER 1
#define EDGE0 2
#define EDGE1 4
#define EDGE2 8
#define FRONTIER_SHIFT 4
#define FRONTIER_MASK  7
#define CELLNO_SHIFT 7

layout (triangles) in;
layout (line_strip, max_vertices = 2) out;

layout (std140) uniform Clipping
{
    float clipping_zcoord;
    float clipping_zproj;
    int clipping_active;
    float width;
    float height;
};

uniform isamplerBuffer TriangleInfo;

float linearstep (float edge0, float edge1, float x)
{
    return  (x - edge0) / (edge1 - edge0);
}

void emit_interpolation (vec4 p1, vec4 p2)
{
    float r = linearstep (p1.w, p2.w, clipping_zproj);
    gl_Position = mix (p1, p2, r);
    EmitVertex ();
}

void main ()
{    
    vec4 me, other;
    float t[3], s = 0.0;
    int v;

    for (int i = 0; i < gl_in.length(); i++) {
        t[i] = step (clipping_zproj, gl_in[i].gl_Position.w);
        s += t[i];
    }

    // triangle is completely behind | in front of the z-plane -> drop!
    if (s == 0.0 || s == 3.0)
        return;

    // triangle intersects z plane (not all points are on the same side)
    int isInner = texelFetch (TriangleInfo, gl_PrimitiveIDIn).r & ISINNER;
    if (isInner != 0) // triangle doesn't belong to surface -> drop!
        return;

    // find v, the "divergent" vertex
    for (v = 0; v < 3; v++)
        if (s == 1.0) {
            if (t[v] == 1.0)
                break; // found
        } else { // s == 2.0
            if (t[v] == 0.0)
                break; // found
        }
        
    me = gl_in[v].gl_Position;

    // the following interpolation works, no matter if v.w is smaller or greater than zplane
    other = gl_in[(v + 1) % 3].gl_Position; // First point
    emit_interpolation (me, other);
            
    other = gl_in[(v + 2) % 3].gl_Position; // Second point
    emit_interpolation (me, other);
}
