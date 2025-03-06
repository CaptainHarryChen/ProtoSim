#version 330 core
out vec4 FragColor;

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
};

const int nLights = 4;
uniform vec3 lightPos[nLights];
uniform vec3 lightColor[nLights];
uniform sampler2D depthMap2D[nLights];
uniform mat4 lightSpaceMatrix[nLights];
uniform bool lightsOn[nLights];
uniform bool enableShadow;

uniform vec3 viewPos;
uniform vec3 albedoIn;
uniform float metallicIn;
uniform float roughnessIn;
uniform float aoIn;
const float PI = 3.14159265359;

vec2 gridSamplingDisk[] = vec2[]
(
    vec2(-2., 2.), vec2(0., 2.82), vec2(2., 2.),
    vec2(-1., 1.), vec2(0., 1.41), vec2(1., 1.),
    vec2(-2.82, 0.), vec2(-1.41, 0.), vec2(0., 0.), vec2(1.41, 0.), vec2(2.82, 0.),
    vec2(-1., -1.), vec2(0., -1.41), vec2(1., -1.),
    vec2(-2., -2.), vec2(0., -2.82), vec2(2., -2.)
);

float ShadowCalculation(vec4 fragPosLightSpace, int i)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    if (projCoords.z > 1.0) return 0.0;
    float closestDepth;
    if (i == 0) closestDepth = texture(depthMap2D[0], projCoords.xy).r;
    if (i == 1) closestDepth = texture(depthMap2D[1], projCoords.xy).r;
    if (i == 2) closestDepth = texture(depthMap2D[2], projCoords.xy).r;
    if (i == 3) closestDepth = texture(depthMap2D[3], projCoords.xy).r;
    float currentDepth = projCoords.z;
    vec3 normal = normalize(Normal);
    vec3 lightDir = normalize(lightPos[i] - FragPos);
    // float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    float bias = 0.0002;
    float shadow = 0.0;
    vec2 texelSize;
    if (i == 0) texelSize = 1.0 / textureSize(depthMap2D[0], 0);
    if (i == 1) texelSize = 1.0 / textureSize(depthMap2D[1], 0);
    if (i == 2) texelSize = 1.0 / textureSize(depthMap2D[2], 0);
    if (i == 3) texelSize = 1.0 / textureSize(depthMap2D[3], 0);
    for (int k = 0; k < 17; ++k)
    {
        float pcfDepth;
        if (i == 0) pcfDepth = texture(depthMap2D[0], projCoords.xy + gridSamplingDisk[k] * texelSize).r;
        if (i == 1) pcfDepth = texture(depthMap2D[1], projCoords.xy + gridSamplingDisk[k] * texelSize).r;
        if (i == 2) pcfDepth = texture(depthMap2D[2], projCoords.xy + gridSamplingDisk[k] * texelSize).r;
        if (i == 3) pcfDepth = texture(depthMap2D[3], projCoords.xy + gridSamplingDisk[k] * texelSize).r;
        shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
    }
    shadow /= 17.0;
    return shadow;
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
    if (!lightsOn[0] && !lightsOn[1] && !lightsOn[2] && !lightsOn[3])
    {
        FragColor = vec4(0., 0., 0., 1.);
        return;
    }
    vec3 albedo = pow(albedoIn, vec3(2.2));
    if (!gl_FrontFacing) albedo = pow(vec3(1.0), vec3(2.2));
    float metallic = metallicIn;
    float roughness = roughnessIn;
    float ao = aoIn;

    vec3 N = Normal;
    vec3 V = normalize(viewPos - FragPos);
    if (!gl_FrontFacing) N = -N;

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < nLights; ++i)
    {
        if (lightsOn[i] == false) continue;
        // calculate per-light radiance
        vec3 L = normalize(lightPos[i] - FragPos);
        vec3 H = normalize(V + L);
        //float distance = length(lightPos - FragPos);
        //float attenuation = 1.0 / (distance * distance);
        //vec3 radiance = lightColor * attenuation;
        vec3 radiance = lightColor[i];


        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001; // + 0.0001 to prevent divide by zero
        vec3 specular = numerator / denominator;

        // kS is equal to Fresnel
        vec3 kS = F;
        // for energy conservation, the diffuse and specular light can't
        // be above 1.0 (unless the surface emits light); to preserve this
        // relationship the diffuse component (kD) should equal 1.0 - kS.
        vec3 kD = vec3(1.0) - kS;
        // multiply kD by the inverse metalness such that only non-metals 
        // have diffuse lighting, or a linear blend if partly metal (pure metals
        // have no diffuse light).
        kD *= 1.0 - metallic;

        // scale light by NdotL
        float NdotL = max(dot(N, L), 0.0);
        vec4 FragPosLightSpace = lightSpaceMatrix[i] * vec4(FragPos, 1.0);
        float shadow = 0.0;
        if(enableShadow) shadow = ShadowCalculation(FragPosLightSpace, i);
        // add to outgoing radiance Lo
        Lo += (kD * albedo / PI + specular) * (1.0 - shadow) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
        //Lo += (kD * albedo / PI + specular) * (1.0 - shadow) * radiance;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again

    }

    // ambient lighting (note that the next IBL tutorial will replace 
    // this ambient lighting with environment lighting).
    vec3 ambient = vec3(0.5) * albedo * ao;
    //if (!lightsOn[0] && !lightsOn[1] && !lightsOn[2] && !lightsOn[3]) ambient = vec3(0.0);

    const float self_emission = 10.0f;
    vec3 color = ambient + Lo * self_emission;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, 1.0);
    //FragColor = vec4(Normal, 1.0);

}