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
#define FRONTIER0 16
#define FRONTIER1 32
#define FRONTIER2 64
#define CELLNO_SHIFT 7

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

uniform isamplerBuffer TriangleInfo;
uniform isamplerBuffer RGBAColors;

in vec3 origPos[];

flat out vec4 theColor;
noperspective out vec3 dist;
flat out vec3 frontier;

void main ()
{    
    int info, ind, col;

    if (clipping_active > 0) {
        float s = 0.0;

        // First check if triangle should be discarded
        for (int i = 0; i < gl_in.length(); i++)
            s += step (clipping_zproj, gl_in[i].gl_Position.w);

        if (s == 0.0) // All points are in front of zplane
            return;
    }

    info = texelFetch (TriangleInfo, gl_PrimitiveIDIn).r;
    ind = info >> CELLNO_SHIFT; // cell no.

    frontier = vec3 ((info & FRONTIER0), (info & FRONTIER1), (info & FRONTIER2));

    col = texelFetch (RGBAColors, ind).r;

    theColor = unpackUnorm4x8 (uint (col));

    // Lighting
    vec3 edge_1 = origPos[1] - origPos[0];
    vec3 edge_2 = origPos[2] - origPos[0];
    vec3 faceNormal = normalize (cross (edge_2, edge_1));
    vec3 lightDir = normalize (vec3 (1.0, 1.0, 1.0));
    //float diff = abs (dot (faceNormal, lightDir));
    float diff = max (dot (faceNormal, lightDir), 0.0);
    theColor = vec4 (mix (0.4, 1.0, diff) * theColor.xyz, theColor.w);

    // Compute barycentric distances
    vec2 display = vec2 (width, height);
    vec2 p0 = display * gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w;
    vec2 p1 = display * gl_in[1].gl_Position.xy / gl_in[1].gl_Position.w;
    vec2 p2 = display * gl_in[2].gl_Position.xy / gl_in[2].gl_Position.w;
    vec2 v0 = p2 - p1;
    vec2 v1 = p2 - p0;
    vec2 v2 = p1 - p0;
    float area = abs (v1.x * v2.y - v1.y * v2.x);

    int edge0 = (info & EDGE0) << 9;
    int edge1 = (info & EDGE1) << 8;
    int edge2 = (info & EDGE2) << 7;

    // vertex passthrough
    gl_Position = gl_in[0].gl_Position;
    dist = vec3 (edge0, area / length (v0), edge2);
    EmitVertex ();

    gl_Position = gl_in[1].gl_Position;
    dist = vec3 (edge0, edge1, area / length (v1));
    EmitVertex ();

    gl_Position = gl_in[2].gl_Position;
    dist = vec3 (area / length (v2), edge1, edge2);
    EmitVertex ();
}
