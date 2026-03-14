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

    vec4 topClip    = u_projMatrix * vec4(topSphereCenter, 1.0);
    vec4 bottomClip = u_projMatrix * vec4(bottomSphereCenter, 1.0);
    vec2 topNDC    = topClip.xy / topClip.w;
    vec2 bottomNDC = bottomClip.xy / bottomClip.w;

    float projScale = u_projMatrix[1][1];
    float topRadiusScreen =
        radius * projScale / -topSphereCenter.z;
    float bottomRadiusScreen =
        radius * projScale / -bottomSphereCenter.z;
    float maxRadiusScreen = max(topRadiusScreen, bottomRadiusScreen);
    vec2 dir = normalize(bottomNDC - topNDC);
    vec2 normal = vec2(-dir.y, dir.x);

    vec2 p1 = topNDC    + normal * maxRadiusScreen - dir * maxRadiusScreen;
    vec2 p2 = topNDC    - normal * maxRadiusScreen - dir * maxRadiusScreen;
    vec2 p3 = bottomNDC + normal * maxRadiusScreen + dir * maxRadiusScreen;
    vec2 p4 = bottomNDC - normal * maxRadiusScreen + dir * maxRadiusScreen;

    float minZ = min(topClip.z / topClip.w, bottomClip.z / bottomClip.w);

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = vec4(p1, minZ, 1.0);
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = vec4(p2, minZ, 1.0);
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = vec4(p3, minZ, 1.0);
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    v_capsuleCenterView = capsuleCenterViewPos;
    v_capsuleAxisView = capsuleAxisView;
    v_radiusView = radius;
    v_halfHeightView = halfHeight;
    gl_Position = vec4(p4, minZ, 1.0);
    FragPos = gs_in[0].FragPos;
    albedoIn = gs_in[0].albedoIn;
    EmitVertex();

    EndPrimitive();
}
