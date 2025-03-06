#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;

out vec3 Color;

void main()
{
    Color = aColor;
    vec4 Pos = vec4(aPos.xy, 1.0, 1.0);
    gl_Position = Pos.xyww;
}