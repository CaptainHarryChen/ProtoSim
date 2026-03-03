#version 330 core

const int nLights = 4;

uniform vec3 lightPos[nLights];
uniform vec3 lightColor[nLights];
uniform bool lightsOn[nLights];
uniform mat4 model;
uniform mat4 view;
uniform vec3 viewPos;
uniform mat4 u_projMatrix;
uniform mat4 u_invProjMatrix;
uniform vec4 u_viewport;

in vec3 v_capsuleCenterView;
in vec3 v_capsuleAxisView;
in float v_radiusView;
in float v_halfHeightView;

const float PI = 3.14159265359;

uniform float roughnessIn;
uniform float metallicIn;

in VS_OUT
{
    vec3 FragPos;
    vec3 albedoIn;
};

layout(location = 0) out vec4 outputF;

float LARGE_FLOAT() { return 1e25; }

vec3 fragmentViewPosition(vec4 viewport, vec2 depthRange, mat4 invProjMat, vec4 fragCoord)
{
    vec4 ndcPos;
    ndcPos.xy = ((2.0 * fragCoord.xy) - (2.0 * viewport.xy)) / (viewport.zw) - 1;
    ndcPos.z = (2.0 * fragCoord.z - depthRange.x - depthRange.y) / (depthRange.y - depthRange.x);
    ndcPos.w = 1.0;

    vec4 clipPos = ndcPos / fragCoord.w;
    vec4 eyePos = invProjMat * clipPos;
    return eyePos.xyz / eyePos.w;
}

float fragDepthFromView(mat4 projMat, vec2 depthRange, vec3 viewPoint)
{
    vec4 clipPos = projMat * vec4(viewPoint, 1.);
    float z_ndc = clipPos.z / clipPos.w;
    float depth = (((depthRange.y - depthRange.x) * z_ndc) + depthRange.x + depthRange.y) / 2.0;
    return depth;
}

bool rayCylinderIntersection(vec3 rayStart, vec3 rayDir, vec3 cylinderCenter, vec3 cylinderAxis, float radius, float halfHeight,
                             out float tHit, out vec3 pHit, out vec3 nHit)
{
    vec3 localStart = rayStart - cylinderCenter;

    vec3 axis = normalize(cylinderAxis);
    vec3 u = axis;

    vec3 v = rayDir;
    vec3 d = localStart;

    float dotUV = dot(u, v);
    float dotUD = dot(u, d);

    vec3 vPerp = v - dotUV * u;
    vec3 dPerp = d - dotUD * u;

    float a = dot(vPerp, vPerp);
    float b = 2.0 * dot(vPerp, dPerp);
    float c = dot(dPerp, dPerp) - radius * radius;

    if (a < 1e-8)
    {
        tHit = LARGE_FLOAT();
        return false;
    }

    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0)
    {
        tHit = LARGE_FLOAT();
        return false;
    }

    float sqrtDisc = sqrt(disc);
    float t1 = (-b - sqrtDisc) / (2.0 * a);
    float t2 = (-b + sqrtDisc) / (2.0 * a);

    float tCyl = LARGE_FLOAT();
    vec3 pCyl;
    bool hitCyl = false;

    if (t1 > 0.001)
    {
        vec3 p = rayStart + t1 * rayDir;
        vec3 localP = p - cylinderCenter;
        float h = dot(localP, axis);
        if (abs(h) <= halfHeight)
        {
            tCyl = t1;
            pCyl = p;
            hitCyl = true;
        }
    }

    if (!hitCyl && t2 > 0.001)
    {
        vec3 p = rayStart + t2 * rayDir;
        vec3 localP = p - cylinderCenter;
        float h = dot(localP, axis);
        if (abs(h) <= halfHeight)
        {
            tCyl = t2;
            pCyl = p;
            hitCyl = true;
        }
    }

    if (hitCyl)
    {
        tHit = tCyl;
        pHit = pCyl;
        vec3 localP = pCyl - cylinderCenter;
        vec3 radial = localP - dot(localP, axis) * axis;
        nHit = normalize(radial);
        return true;
    }

    tHit = LARGE_FLOAT();
    return false;
}

bool raySphereIntersection(vec3 rayStart, vec3 rayDir, vec3 sphereCenter, float sphereRad,
                           out float tHit, out vec3 pHit, out vec3 nHit)
{
    vec3 o = rayStart - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(o, rayDir);
    float c = dot(o, o) - sphereRad * sphereRad;
    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0)
    {
        tHit = LARGE_FLOAT();
        pHit = vec3(777, 777, 777);
        nHit = vec3(777, 777, 777);
        return false;
    }
    tHit = (-b - sqrt(disc)) / (2.0 * a);
    pHit = rayStart + tHit * rayDir;
    nHit = normalize(pHit - sphereCenter);
    return true;
}

bool rayCapsuleIntersection(vec3 rayStart, vec3 rayDir, vec3 capsuleCenter, vec3 capsuleAxis, float radius, float halfHeight,
                            out float tHit, out vec3 pHit, out vec3 nHit)
{
    vec3 axis = normalize(capsuleAxis);

    float tCyl;
    vec3 pCyl, nCyl;
    bool hitCyl = rayCylinderIntersection(rayStart, rayDir, capsuleCenter, axis, radius, halfHeight, tCyl, pCyl, nCyl);

    vec3 topCenter = capsuleCenter + axis * halfHeight;
    vec3 bottomCenter = capsuleCenter - axis * halfHeight;

    float tTop, tBottom;
    vec3 pTop, pBottom, nTop, nBottom;
    bool hitTop = raySphereIntersection(rayStart, rayDir, topCenter, radius, tTop, pTop, nTop);
    bool hitBottom = raySphereIntersection(rayStart, rayDir, bottomCenter, radius, tBottom, pBottom, nBottom);

    if (hitTop)
    {
        vec3 localP = pTop - topCenter;
        if (dot(localP, axis) < 0.0)
            hitTop = false;
    }
    if (hitBottom)
    {
        vec3 localP = pBottom - bottomCenter;
        if (dot(localP, axis) > 0.0)
            hitBottom = false;
    }

    float tBest = LARGE_FLOAT();
    vec3 pBest, nBest;
    bool hit = false;

    if (hitCyl && tCyl < tBest)
    {
        tBest = tCyl;
        pBest = pCyl;
        nBest = nCyl;
        hit = true;
    }
    if (hitTop && tTop < tBest)
    {
        tBest = tTop;
        pBest = pTop;
        nBest = nTop;
        hit = true;
    }
    if (hitBottom && tBottom < tBest)
    {
        tBest = tBottom;
        pBest = pBottom;
        nBest = nBottom;
        hit = true;
    }

    tHit = tBest;
    pHit = pBest;
    nHit = nBest;
    return hit;
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main()
{
    vec2 depthRange = vec2(gl_DepthRange.near, gl_DepthRange.far);
    vec3 viewRay = fragmentViewPosition(u_viewport, depthRange, u_invProjMatrix, gl_FragCoord);

    float radius = v_radiusView;
    float halfHeight = v_halfHeightView;
    vec3 axis = v_capsuleAxisView;

    float tHit;
    vec3 pHit, nHit;
    bool hit = rayCapsuleIntersection(vec3(0., 0., 0), viewRay, v_capsuleCenterView, axis, radius, halfHeight, tHit, pHit, nHit);

    if (!hit || tHit >= LARGE_FLOAT())
        discard;

    if (!lightsOn[0] && !lightsOn[1] && !lightsOn[2] && !lightsOn[3])
    {
        outputF = vec4(0., 0., 0., 1.);
        return;
    }

    float depth = fragDepthFromView(u_projMatrix, depthRange, pHit);
    gl_FragDepth = depth;

    mat3 mv_inv = mat3(inverse(view * model));
    vec3 N = mv_inv * normalize(nHit);
    vec3 V = normalize(viewPos - FragPos);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedoIn, metallicIn);
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < nLights; ++i)
    {
        if (lightsOn[i] == false)
            continue;
        vec3 L = normalize(lightPos[i] - FragPos);
        vec3 H = normalize(V + L);
        vec3 radiance = lightColor[i];

        float NDF = DistributionGGX(N, H, roughnessIn);
        float G = GeometrySmith(N, V, L, roughnessIn);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallicIn;
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedoIn / PI + specular) * radiance * NdotL;
    }

    vec3 ambient = vec3(0.1) * albedoIn;
    vec3 color = ambient + Lo * 2.0f;

    color = pow(color, vec3(1.0 / 2.2));
    outputF = vec4(color, 1.0);
}
