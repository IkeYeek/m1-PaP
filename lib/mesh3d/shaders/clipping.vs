#version 400 core

layout (location = 0) in vec2 pos;

layout (std140) uniform Matrices
{
    mat4 mvp;
    mat4 ortho;
    mat4 vp_unclipped;
    mat4 mvp_unclipped;
};

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
    gl_Position = vp_unclipped * vec4 (pos, clipping_zcoord, 1.0);
}
