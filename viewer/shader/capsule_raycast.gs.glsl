#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in VS_OUT
{
    vec3 FragPos;
    vec3 albedoIn;
} gs_in[];

out VS_OUT
{
    vec3 FragPos;
    vec3 albedoIn;
};

uniform mat4 u_projMatrix;
uniform mat4 model;
uniform mat4 view;
uniform float u_radius;
uniform float u_halfHeight;

out vec3 v_capsuleCenterView;
out vec3 v_capsuleAxisView;
out float v_radiusView;
out float v_halfHeightView;

void buildTangentBasis(vec3 unitNormal, out vec3 basisX, out vec3 basisY)
{
    basisX = vec3(1., 0., 0.);
    basisX -= dot(basisX, unitNormal) * unitNormal;
    if (abs(basisX.x) < 0.1)
    {
        basisX = vec3(0., 1., 0.);
        basisX -= dot(basisX, unitNormal) * unitNormal;
    }
    basisX = normalize(basisX);
    basisY = normalize(cross(unitNormal, basisX));
}

void main()
{
    float radius = u_radius;
    float halfHeight = u_halfHeight;

    vec3 capsuleCenterViewPos = gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w;

    vec3 capsuleAxisWorld = vec3(0.0, 1.0, 0.0);
    vec3 capsuleAxisView = normalize(mat3(view * model) * capsuleAxisWorld);

    vec3 topSphereCenter = capsuleCenterViewPos + capsuleAxisView * halfHeight;
    vec3 bottomSphereCenter = capsuleCenterViewPos - capsuleAxisView * halfHeight;

    vec3 dirToCam = normalize(-capsuleCenterViewPos);
    vec3 basisX, basisY;
    buildTangentBasis(dirToCam, basisX, basisY);

    vec4 topCenter = u_projMatrix * vec4(topSphereCenter + dirToCam * radius, 1.0);
    vec4 bottomCenter = u_projMatrix * vec4(bottomSphereCenter + dirToCam * radius, 1.0);

    vec4 dx = u_projMatrix * vec4(basisX * radius, 0.0);
    vec4 dy = u_projMatrix * vec4(basisY * radius, 0.0);

    vec4 topP1 = topCenter - dx - dy;
    vec4 topP2 = topCenter + dx - dy;
    vec4 topP3 = topCenter - dx + dy;
    vec4 topP4 = topCenter + dx + dy;

    vec4 bottomP1 = bottomCenter - dx - dy;
    vec4 bottomP2 = bottomCenter + dx - dy;
    vec4 bottomP3 = bottomCenter - dx + dy;
    vec4 bottomP4 = bottomCenter + dx + dy;

    vec4 vertices[8] = vec4[](topP1, topP2, topP3, topP4, bottomP1, bottomP2, bottomP3, bottomP4);

    vec2 minScreen = vertices[0].xy / vertices[0].w;
    vec2 maxScreen = vertices[0].xy / vertices[0].w;
    for (int i = 1; i < 8; ++i)
    {
        vec2 screen = vertices[i].xy / vertices[i].w;
        minScreen = min(minScreen, screen);
        maxScreen = max(maxScreen, screen);
    }

    float minZ = vertices[0].z / vertices[0].w;
    for (int i = 1; i < 8; ++i)
    {
        float z = vertices[i].z / vertices[i].w;
        minZ = min(minZ, z);
    }

    vec4 p1 = vec4(minScreen.x, minScreen.y, minZ, 1.0);
    vec4 p2 = vec4(maxScreen.x, minScreen.y, minZ, 1.0);
    vec4 p3 = vec4(minScreen.x, maxScreen.y, minZ, 1.0);
    vec4 p4 = vec4(maxScreen.x, maxScreen.y, minZ, 1.0);

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = p1;
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = p2;
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = p3;
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = p4;
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    EndPrimitive();
}
