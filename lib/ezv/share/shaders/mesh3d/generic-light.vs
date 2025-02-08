#version 400 core

layout (location = 0) in vec3 pos;

layout (std140) uniform Matrices
{
    mat4 mvp;
    mat4 ortho;
    mat4 vp_unclipped;
    mat4 mvp_unclipped;
    mat4 mv;
};

out vec3 origPos;

void main()
{
    origPos = vec3 (mv * vec4 (pos, 1.0));
    gl_Position = mvp * vec4 (pos, 1.0);
}
