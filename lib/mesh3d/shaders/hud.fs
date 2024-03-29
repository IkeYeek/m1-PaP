#version 400 core

in vec2 TexCoord;
uniform sampler2D digitTexture;

out vec4 FragColor;

void main()
{
    FragColor = texture (digitTexture, TexCoord);
}
