#version 400 core

#define THICKNESS 1.0

layout (std140) uniform CustomColors
{
    vec4 mesh_color;
    vec4 cut_color;
};

flat in vec4 theColor;
noperspective in vec3 dist;
flat in vec3 frontier;

out vec4 FragColor;

float min3 (vec3 v)
{
    return min (v.x, min (v.y, v.z));
}

void main ()
{
    float d = min3 (dist);

    if (d < THICKNESS) {
        vec3 m = step (dist, vec3(THICKNESS, THICKNESS, THICKNESS));
        if (dot (m, frontier) > 0.0)
            FragColor = mix (theColor, cut_color, (1.0 - 0.4 * theColor.w) * (THICKNESS - d));
        else
            FragColor = mix (theColor, mesh_color, (1.0 - 0.5 * theColor.w) * (THICKNESS - d));
        
        return;
    }

    if (theColor.w == 0.0)
        discard;

    FragColor = theColor;
}
