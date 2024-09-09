#version 400 core

layout (location = 0) in vec3 pos;

layout (std140) uniform Matrices
{
    mat4 mvp;
    mat4 ortho;
    mat4 vp_unclipped;
    mat4 mvp_unclipped;
};

void main()
{
    gl_Position = mvp * vec4 (pos, 1.0);
}
