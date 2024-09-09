#version 400 core

noperspective in vec2 coord;

out vec2 FragColor;

void main()
{
    FragColor = coord;
}
