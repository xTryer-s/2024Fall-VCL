#version 410 core

layout(location = 0) in  vec3 v_Position;
layout(location = 1) in  vec3 v_Normal;
layout(location = 2) in  vec2 v_TexCoord;

layout(location = 0) out vec4 f_Color;

struct Light {
    vec3  Intensity;
    vec3  Direction;   // For spot and directional lights.
    vec3  Position;    // For point and spot lights.
    float CutOff;      // For spot lights.
    float OuterCutOff; // For spot lights.
};

layout(std140) uniform PassConstants {
    mat4  u_Projection;
    mat4  u_View;
    vec3  u_ViewPosition;
    vec3  u_AmbientIntensity;
    Light u_Lights[4];
    int   u_CntPointLights;
    int   u_CntSpotLights;
    int   u_CntDirectionalLights;
};

uniform float u_AmbientScale;
uniform bool  u_UseBlinn;
uniform float u_Shininess;
uniform bool  u_UseGammaCorrection;
uniform int   u_AttenuationOrder;
uniform float u_BumpMappingBlend;

uniform sampler2D u_DiffuseMap;
uniform sampler2D u_SpecularMap;
uniform sampler2D u_HeightMap;

vec3 Shade(vec3 lightIntensity, vec3 lightDir, vec3 normal, vec3 viewDir, vec3 diffuseColor, vec3 specularColor, float shininess) {
    // your code here:

    vec3 normalized_lightdir = normalize(lightDir);
    vec3 normalized_normaldir = normalize(normal);
    vec3 normalized_viewdir=normalize(viewDir);

    // diffuse_light
    float diffuse_cos = max(0.0,dot(normalized_lightdir,normalized_normaldir));
    vec3 diffuse_light = diffuseColor*(diffuse_cos*lightIntensity);

    //specular_light
    //Phong:
    vec3 reflect_light_dir_Phong = dot(normalized_lightdir,normalized_normaldir)*normalized_normaldir*2.0-normalized_lightdir;
    //Blinn-Phong:
    vec3 normalized_halfdir = normalize(normalized_lightdir+normalized_viewdir);

    float specular_degree1 = pow(max(0.0,dot(reflect_light_dir_Phong,normalized_viewdir)),shininess);
    float specular_degree2 = pow(max(0.0,dot(normalized_halfdir,normalized_normaldir)),shininess);
    
    vec3 specular_light1 = specularColor*specular_degree1*lightIntensity;
    vec3 specular_light2 = specularColor*specular_degree2*lightIntensity;



    return (u_UseBlinn==true)?diffuse_light+specular_light2:diffuse_light+specular_light1;
}

vec3 GetNormal() {
    // Bump mapping from paper: Bump Mapping Unparametrized Surfaces on the GPU
    vec3 vn = normalize(v_Normal);

    // your code here:
    vec3 bumpNormal = vn;
    float offset_ =1.0/512.0;
    float heightL = texture(u_HeightMap, v_TexCoord - vec2(offset_, 0.0)).r;
    float heightR = texture(u_HeightMap, v_TexCoord + vec2(offset_, 0.0)).r;
    float heightD = texture(u_HeightMap, v_TexCoord - vec2(0.0, offset_)).r;
    float heightU = texture(u_HeightMap, v_TexCoord + vec2(0.0, offset_)).r;
    
    float height_dx = (heightR - heightL);
    float height_dy = (heightU - heightD);

    bumpNormal=normalize(vec3(-height_dx,-height_dy,2.0*offset_));

    return bumpNormal != bumpNormal ? vn : normalize(vn * (1. - u_BumpMappingBlend) + bumpNormal * u_BumpMappingBlend);
}

void main() {
    float gamma          = 2.2;
    vec4  diffuseFactor  = texture(u_DiffuseMap , v_TexCoord).rgba;
    vec4  specularFactor = texture(u_SpecularMap, v_TexCoord).rgba;
    if (diffuseFactor.a < .2) discard;
    vec3  diffuseColor   = u_UseGammaCorrection ? pow(diffuseFactor.rgb, vec3(gamma)) : diffuseFactor.rgb;
    vec3  specularColor  = specularFactor.rgb;
    float shininess      = u_Shininess < 0 ? specularFactor.a * 256 : u_Shininess;
    vec3  normal         = GetNormal();
    vec3  viewDir        = normalize(u_ViewPosition - v_Position);
    // Ambient component.
    vec3  total = u_AmbientIntensity * u_AmbientScale * diffuseColor;
    // Iterate lights.
    for (int i = 0; i < u_CntPointLights; i++) {
        vec3  lightDir     = normalize(u_Lights[i].Position - v_Position);
        float dist         = length(u_Lights[i].Position - v_Position);
        float attenuation  = 1. / (u_AttenuationOrder == 2 ? dist * dist : (u_AttenuationOrder == 1  ? dist : 1.));
        total             += Shade(u_Lights[i].Intensity, lightDir, normal, viewDir, diffuseColor, specularColor, shininess) * attenuation;
    }
    for (int i = u_CntPointLights + u_CntSpotLights; i < u_CntPointLights + u_CntSpotLights + u_CntDirectionalLights; i++) {
        total += Shade(u_Lights[i].Intensity, u_Lights[i].Direction, normal, viewDir, diffuseColor, specularColor, shininess);
    }
    // Gamma correction.
    f_Color = vec4(u_UseGammaCorrection ? pow(total, vec3(1. / gamma)) : total, 1.);
}
