#version 330 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    //vec4 FragPosLightSpace;
} gs_in[];

out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    //vec4 FragPosLightSpace;
};

void PassThrough(int i)
{
    FragPos = gs_in[i].FragPos;
    //Normal = gs_in[i].Normal;
    TexCoords = gs_in[i].TexCoords;
    //FragPosLightSpace = gs_in[i].FragPosLightSpace;
    gl_Position = gl_in[i].gl_Position;
    EmitVertex();
}

vec3 GenerateSurfaceNormal()
{
    vec3 a = vec3(gs_in[1].FragPos) - vec3(gs_in[0].FragPos);
    vec3 b = vec3(gs_in[2].FragPos) - vec3(gs_in[0].FragPos);
    return normalize(cross(a, b));
}

void main()
{
    Normal = GenerateSurfaceNormal();
    PassThrough(0);
    PassThrough(1);
    PassThrough(2);
    EndPrimitive();
}
