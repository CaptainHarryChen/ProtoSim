#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    //vec4 FragPosLightSpace;
};

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
//uniform mat4 lightSpaceMatrix;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * model * vec4(FragPos, 1.0);
    //FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
}