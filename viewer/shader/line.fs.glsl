#version 330 core
out vec4 FragColor;
uniform vec3 color;
uniform bool lightOn;

void main()
{
    FragColor = vec4(color, 1.);
    if (lightOn == false) FragColor = vec4(0., 0., 0., 1.);
}