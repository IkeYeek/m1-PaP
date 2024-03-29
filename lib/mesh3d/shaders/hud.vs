#version 400 core

#define MAX_DIGITS 128

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 texCoord;

layout (std140) uniform Matrices
{
    mat4 mvp;
    mat4 ortho;
    mat4 vp_unclipped;
    mat4 mvp_unclipped;
};

layout (std140) uniform HudInfo
{
    float digit_width;
    float x_spacing;
    float y_spacing;
};

uniform int digits[MAX_DIGITS];
uniform int line;

out vec2 TexCoord;

void main()
{
    vec2 offset = vec2 (5.0, 5.0);
    int d = digits[gl_InstanceID];

    offset.x += gl_InstanceID * x_spacing;
    offset.y += line * y_spacing;

    gl_Position = ortho * vec4 (pos + offset, -1.0, 1.0);
    TexCoord = texCoord + vec2 (d * digit_width, 0.0);
}
