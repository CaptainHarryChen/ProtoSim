#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in VS_OUT
{
    vec3 FragPos;
    vec3 albedoIn;
}gs_in[];

out VS_OUT
{
    vec3 FragPos;
    vec3 albedoIn;
};

uniform mat4 u_projMatrix;
uniform float u_pointRadius;
out vec3 sphereCenterView;

void buildTangentBasis(vec3 unitNormal, out vec3 basisX, out vec3 basisY) {
    basisX = vec3(1., 0., 0.);
    basisX -= dot(basisX, unitNormal) * unitNormal;
    if (abs(basisX.x) < 0.1) {
        basisX = vec3(0., 1., 0.);
        basisX -= dot(basisX, unitNormal) * unitNormal;
    }
    basisX = normalize(basisX);
    basisY = normalize(cross(unitNormal, basisX));
}

void main() {

    //FragPos = vsFragPos;
    float pointRadius = u_pointRadius;

    vec3 dirToCam = normalize(-gl_in[0].gl_Position.xyz);
    vec3 basisX;
    vec3 basisY;
    buildTangentBasis(dirToCam, basisX, basisY);
    vec4 center = u_projMatrix * (gl_in[0].gl_Position + vec4(dirToCam, 0.) * pointRadius);
    vec4 dx = u_projMatrix * (vec4(basisX, 0.) * pointRadius);
    vec4 dy = u_projMatrix * (vec4(basisY, 0.) * pointRadius);
    vec4 p1 = center - dx - dy;
    vec4 p2 = center + dx - dy;
    vec4 p3 = center - dx + dy;
    vec4 p4 = center + dx + dy;

    // Other data to emit
    vec3 sphereCenterViewVal = gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w;

    // Emit the vertices as a triangle strip
    sphereCenterView = sphereCenterViewVal; gl_Position = p1; FragPos = gs_in[0].FragPos; albedoIn = gs_in[0].albedoIn; EmitVertex();
    sphereCenterView = sphereCenterViewVal; gl_Position = p2; FragPos = gs_in[0].FragPos; albedoIn = gs_in[0].albedoIn; EmitVertex();
    sphereCenterView = sphereCenterViewVal; gl_Position = p3; FragPos = gs_in[0].FragPos; albedoIn = gs_in[0].albedoIn; EmitVertex();
    sphereCenterView = sphereCenterViewVal; gl_Position = p4; FragPos = gs_in[0].FragPos; albedoIn = gs_in[0].albedoIn; EmitVertex();

    EndPrimitive();

}