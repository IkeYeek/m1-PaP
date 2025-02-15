#version 400 core

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 texCoord;

layout (std140) uniform Matrices
{
    mat4 mvp;
    mat4 ortho;
    mat4 vp_unclipped;
    mat4 mvp_unclipped;
    mat4 mv;
};

out vec2 TexCoord;

void main()
{
    gl_Position = ortho * vec4 (pos, -1.0, 1.0);
    TexCoord = texCoord;
}
