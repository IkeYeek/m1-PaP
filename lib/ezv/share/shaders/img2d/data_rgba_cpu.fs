#version 400 core

in vec2 TexCoord;
uniform sampler2D cpuTexture;
uniform sampler2D dataTexture;

uniform float dataBrightness;

out vec4 FragColor;

void main()
{
    vec4 dataColor, cpuColor;

    cpuColor = texture (cpuTexture, TexCoord);
    dataColor = texture (dataTexture, TexCoord);

    // Reduce luminance
    dataColor = vec4 (dataColor.xyz * dataBrightness, dataColor.w);

    // Blend
    FragColor = mix (dataColor, vec4 (cpuColor.xyz, 1.0), cpuColor.w);
}
