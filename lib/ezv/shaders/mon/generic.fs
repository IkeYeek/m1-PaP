#version 400 core

in vec2 TexCoord;
uniform sampler2D dataTexture;

out vec4 FragColor;

void main()
{
    FragColor = texture (dataTexture, TexCoord);
}
