#version 410 core

layout(location = 0) in  vec3 v_Position;
layout(location = 1) in  vec3 v_Normal;

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

uniform vec3 u_CoolColor;
uniform vec3 u_WarmColor;

vec3 Shade (vec3 lightDir, vec3 normal) {
    // your code here:
    vec3 normal_lightDir = normalize(lightDir);
    vec3 normal_normalDir = normalize(normal);
    float cos_ = dot(normal_lightDir,normal_normalDir);
    vec3 mix_color = vec3(0);

    float threshold_1=0.25;
    float threshold_2=0.7;
    if(cos_>threshold_2)
    {
        mix_color=u_WarmColor;
    }
    else if(cos_>threshold_1)
    {
        mix_color=u_WarmColor*(1+cos_)/2.0+u_CoolColor*(1-cos_)/2.0;
    }
    else
    {
        mix_color=u_CoolColor;
    }
    return mix_color;
}

void main() {
    // your code here:
    float gamma = 2.2;
    vec3 total = Shade(u_Lights[0].Direction, v_Normal);
    f_Color = vec4(pow(total, vec3(1. / gamma)), 1.);
}
