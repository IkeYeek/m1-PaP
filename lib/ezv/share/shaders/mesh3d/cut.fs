#version 400 core

layout (std140) uniform CustomColors
{
    vec4 mesh_color;
    vec4 cut_color;
};

out vec4 FragColor;

void main()
{
    FragColor = cut_color;
}
