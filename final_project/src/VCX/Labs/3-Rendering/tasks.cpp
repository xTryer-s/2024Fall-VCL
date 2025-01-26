#include "Labs/3-Rendering/tasks.h"
#include<cmath>
namespace VCX::Labs::Rendering {

    glm::vec4 GetTexture(Engine::Texture2D<Engine::Formats::RGBA8> const & texture, glm::vec2 const & uvCoord) {
        if (texture.GetSizeX() == 1 || texture.GetSizeY() == 1) return texture.At(0, 0);
        glm::vec2 uv      = glm::fract(uvCoord);
        uv.x              = uv.x * texture.GetSizeX() - .5f;
        uv.y              = uv.y * texture.GetSizeY() - .5f;
        std::size_t xmin  = std::size_t(glm::floor(uv.x) + texture.GetSizeX()) % texture.GetSizeX();
        std::size_t ymin  = std::size_t(glm::floor(uv.y) + texture.GetSizeY()) % texture.GetSizeY();
        std::size_t xmax  = (xmin + 1) % texture.GetSizeX();
        std::size_t ymax  = (ymin + 1) % texture.GetSizeY();
        float       xfrac = glm::fract(uv.x), yfrac = glm::fract(uv.y);
        return glm::mix(glm::mix(texture.At(xmin, ymin), texture.At(xmin, ymax), yfrac), glm::mix(texture.At(xmax, ymin), texture.At(xmax, ymax), yfrac), xfrac);
    }

    glm::vec4 GetAlbedo(Engine::Material const & material, glm::vec2 const & uvCoord) {
        glm::vec4 albedo       = GetTexture(material.Albedo, uvCoord);
        glm::vec3 diffuseColor = albedo;
        return glm::vec4(glm::pow(diffuseColor, glm::vec3(2.2)), albedo.w);
    }

    /******************* 1. Ray-triangle intersection *****************/
    bool IntersectTriangle(Intersection & output, Ray const & ray, glm::vec3 const & p1, glm::vec3 const & p2, glm::vec3 const & p3) {
        // your code here

        glm::vec3 edge12 = p2 - p1;
        glm::vec3 edge13 = p3 - p1;

        glm::vec3 normal_dir = normalize(ray.Direction);
        glm::vec3 origin_point     = ray.Origin;

        glm::vec3 pvec = glm::cross(normal_dir, edge13);
        double    det  = glm::dot(edge12, pvec);

        if (abs(det) < 0.00005)
        {
            return false;
        }
        double inv_det = 1.0 / det;

        glm::vec3 tvec = origin_point - p1;
        double    u_   = glm::dot(tvec, pvec)*inv_det;
        if (u_<0.0 || u_>1.0)
        {
            return false;
        }

        glm::vec3 qvec = glm::cross(tvec, edge12);
        double    v_   = glm::dot(normal_dir, qvec)*inv_det;
        if (v_<0.0 || u_ + v_>1.0)
        {
            return false;
        }

        double t_ = glm::dot(edge13,qvec)*inv_det;

        output.t = t_;
        output.u = u_;
        output.v = v_;
        return true;
    }

    glm::vec3 RayTrace(const RayIntersector & intersector, Ray ray, int maxDepth, bool enableShadow) {
        glm::vec3 color(0.0f);
        glm::vec3 weight(1.0f);

        for (int depth = 0; depth < maxDepth; depth++) {
            auto rayHit = intersector.IntersectRay(ray);
            if (! rayHit.IntersectState) return color;
            const glm::vec3 pos       = rayHit.IntersectPosition;
            const glm::vec3 n         = rayHit.IntersectNormal;
            const glm::vec3 kd        = rayHit.IntersectAlbedo;
            const glm::vec3 ks        = rayHit.IntersectMetaSpec;
            const float     alpha     = rayHit.IntersectAlbedo.w;
            const float     shininess = rayHit.IntersectMetaSpec.w * 256;

            glm::vec3 result(0.0f);
            /******************* 2. Whitted-style ray tracing *****************/
            // your code here

            for (const Engine::Light & light : intersector.InternalScene->Lights) {
                glm::vec3 l;
                float     attenuation;
                /******************* 3. Shadow ray *****************/
                if (light.Type == Engine::LightType::Point) {
                    l           = light.Position - pos;
                    attenuation = 1.0f / glm::dot(l, l);
                    if (enableShadow) {
                        auto shadowHit_ = intersector.IntersectRay(Ray(pos, l));
                        if (shadowHit_.IntersectState==true && shadowHit_.IntersectAlbedo.w >= 0.2f) {
                            
                            // consider light&scene ray cross point in the back
                            glm::vec3 shadow_pos = shadowHit_.IntersectPosition;
                            glm::vec3 shadow_light_dir = light.Position-shadow_pos;
                            if (glm::dot(shadow_light_dir,l)>0) continue; 
                        }
                    }
                } else if (light.Type == Engine::LightType::Directional) {
                    l           = light.Direction;
                    attenuation = 1.0f;
                    if (enableShadow) {
                        auto shadowHit_ = intersector.IntersectRay(Ray(pos, l));
                        if (shadowHit_.IntersectState==true && shadowHit_.IntersectAlbedo.w >= 0.2f) {
                            continue; 
                        }
                    }
                }

                /******************* 2. Whitted-style ray tracing *****************/
                // your code here

                glm::vec3 normalize_normal = normalize(n);
                glm::vec3 normalize_lightdir = normalize(l);

               // diffuse_light 
                float diff_cos = glm::max(glm::dot(normalize_normal, normalize_lightdir), 0.0f);
                glm::vec3 diffuse_light= kd * diff_cos * light.Intensity * attenuation;

                // specular_light
                // use_blinn-phong
                glm::vec3 normalize_raydir = normalize(ray.Direction);
                glm::vec3 normalize_halfdir = normalize(normalize_raydir+normalize_lightdir);
                float     spec_cos          = pow(glm::max(0.0f,glm::dot(normalize_raydir,normalize_normal)), shininess);

                glm::vec3 specular_light = ks*spec_cos*light.Intensity*attenuation;

                result += (diffuse_light + specular_light);
            }

            // add ambient light
            result += kd * intersector.InternalScene->AmbientIntensity;

            if (alpha < 0.9) {
                // refraction
                // accumulate color
                glm::vec3 R = alpha * glm::vec3(1.0f);
                color += weight * R * result;
                weight *= glm::vec3(1.0f) - R;

                // generate new ray
                ray = Ray(pos, ray.Direction);
            } else {
                // reflection
                // accumulate color
                glm::vec3 R = ks * glm::vec3(0.5f);
                color += weight * (glm::vec3(1.0f) - R) * result;
                weight *= R;

                // generate new ray
                glm::vec3 out_dir = ray.Direction - glm::vec3(2.0f) * n * glm::dot(n, ray.Direction);
                ray               = Ray(pos, out_dir);
            }
        }

        return color;
    }
} // namespace VCX::Labs::Rendering