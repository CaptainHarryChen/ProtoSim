#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 color;

out VS_OUT
{
    vec3 FragPos;
    vec3 albedoIn;
};

uniform mat4 model;
uniform mat4 view;

void main()
{
    gl_Position = view * model * vec4(a_position, 1.0);
    albedoIn = color;
    FragPos = mat3(model) * a_position;
}