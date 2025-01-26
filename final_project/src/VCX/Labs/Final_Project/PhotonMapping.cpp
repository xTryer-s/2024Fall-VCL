#include "Labs/Final_Project/PhotonMapping.h"
#include<cmath>
namespace VCX::Labs::Rendering {

    glm::vec4 p_GetTexture(Engine::Texture2D<Engine::Formats::RGBA8> const & texture, glm::vec2 const & uvCoord) {
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

    glm::vec4 p_GetAlbedo(Engine::Material const & material, glm::vec2 const & uvCoord) {
        glm::vec4 albedo       = p_GetTexture(material.Albedo, uvCoord);
        glm::vec3 diffuseColor = albedo;
        return glm::vec4(glm::pow(diffuseColor, glm::vec3(2.2)), albedo.w);
    }

    /******************* 1. Ray-triangle intersection *****************/
    bool p_IntersectTriangle(p_Intersection & output, Ray const & ray, glm::vec3 const & p1, glm::vec3 const & p2, glm::vec3 const & p3) {
        // your code here

        glm::vec3 edge12 = p2 - p1;
        glm::vec3 edge13 = p3 - p1;

        glm::vec3 normal_dir   = normalize(ray.Direction);
        glm::vec3 origin_point = ray.Origin;

        glm::vec3 pvec = glm::cross(normal_dir, edge13);
        double    det  = glm::dot(edge12, pvec);

        if (abs(det) < 0.00005) {
            return false;
        }
        double inv_det = 1.0 / det;

        glm::vec3 tvec = origin_point - p1;
        double    u_   = glm::dot(tvec, pvec) * inv_det;
        if (u_ < 0.0 || u_ > 1.0) {
            return false;
        }

        glm::vec3 qvec = glm::cross(tvec, edge12);
        double    v_   = glm::dot(normal_dir, qvec) * inv_det;
        if (v_ < 0.0 || u_ + v_ > 1.0) {
            return false;
        }

        double t_ = glm::dot(edge13, qvec) * inv_det;

        output.t = t_;
        output.u = u_;
        output.v = v_;
        return true;
    }

    glm::vec3 RayTraceWithPhotonMapping(const p_RayIntersector & intersector, Ray ray, int maxDepth, const PhotonMapping & photonMapping, int near_photon_nums, float photonRadius,bool enablekd) {
        glm::vec3 color(0.0f);
        glm::vec3 weight(1.0f);

        for (int depth = 0; depth < maxDepth; depth++) {
            auto p_RayHit = intersector.IntersectRay(ray);
            if (! p_RayHit.IntersectState) return color;

           
            const glm::vec3 pos       = p_RayHit.IntersectPosition;
            const glm::vec3 n         = p_RayHit.IntersectNormal;
            const glm::vec3 kd        = p_RayHit.IntersectAlbedo;
            const glm::vec3 ks        = p_RayHit.IntersectMetaSpec;
            const float     alpha     = p_RayHit.IntersectAlbedo.w;
            const float     shininess = p_RayHit.IntersectMetaSpec.w * 256;

            glm::vec3 result(0.0f);

            // Photonmapping estimate radiance
            result += photonMapping.calculate_photon_radiance(p_RayHit,ray, near_photon_nums,photonRadius,enablekd);
            

            // ambient light
            result += kd * intersector.InternalScene->AmbientIntensity;

            // Refraction and Reflection
            if (alpha < 0.9f || p_RayHit.IntersectMode == Engine::BlendMode::Transparent ) {   
                // Refraction
                glm::vec3 R = alpha * glm::vec3(1.0f);
                color += weight * R * result;
                weight *= glm::vec3(1.0f) - R;
                ray = Ray(pos, ray.Direction);
            } else {
                // Reflection
                glm::vec3 R = ks * glm::vec3(0.5f);
                color += weight * (glm::vec3(1.0f) - R) * result;
                weight *= R;
                glm::vec3 outDir = ray.Direction - 2.0f * n * glm::dot(n, ray.Direction);
                ray              = Ray(pos, outDir);
            }
        }

        return color;
    }

} // namespace VCX::Labs::Rendering