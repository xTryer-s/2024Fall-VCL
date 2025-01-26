#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/random.hpp>
#include <vector>
#include "Labs/Final_Project/Photon.h"
#include "Engine/Scene.h"
#include "Labs/Final_Project/Ray.h"
#include <numeric>
#include <spdlog/spdlog.h>
#include <queue>
#include<vector>
#include<algorithm>
#include<random>

namespace VCX::Labs::Rendering {

    constexpr float p_EPS1 = 1e-2f; // distance to prevent self-intersection
    constexpr float p_EPS2 = 1e-8f; // angle for parallel judgement
    constexpr float p_EPS3 = 1e-4f; // relative distance to enlarge kdtree

    glm::vec4 p_GetTexture(Engine::Texture2D<Engine::Formats::RGBA8> const & texture, glm::vec2 const & uvCoord);

    glm::vec4 p_GetAlbedo(Engine::Material const & material, glm::vec2 const & uvCoord);

    struct p_Intersection {
        float t, u, v; // ray parameter t, barycentric coordinates (u, v)
    };

    bool p_IntersectTriangle(p_Intersection & output, Ray const & ray, glm::vec3 const & p1, glm::vec3 const & p2, glm::vec3 const & p3);

    struct p_RayHit {
        bool              IntersectState;
        Engine::BlendMode IntersectMode;
        glm::vec3         IntersectPosition;
        glm::vec3         IntersectNormal;
        glm::vec4         IntersectAlbedo;   // [Albedo   (vec3), Alpha     (float)]
        glm::vec4         IntersectMetaSpec; // [Specular (vec3), Shininess (float)]
    };

    struct p_TrivialRayIntersector {
        Engine::Scene const * InternalScene = nullptr;

        p_TrivialRayIntersector() = default;

        void InitScene(Engine::Scene const * scene) {
            printf("begin init scene\n");
            InternalScene = scene;
        }

        p_RayHit IntersectRay(Ray const & ray) const {
            p_RayHit result;
            if (! InternalScene) {
                spdlog::warn("VCX::Labs::Rendering::RayIntersector::IntersectRay(..): uninitialized intersector.");
                result.IntersectState = false;
                return result;
            }
            int          modelIdx, meshIdx;
            p_Intersection its;
            float        tmin     = 1e7, umin, vmin;
            int          maxmodel = InternalScene->Models.size();
            for (int i = 0; i < maxmodel; ++i) {
                auto const & model  = InternalScene->Models[i];
                int          maxidx = model.Mesh.Indices.size();
                for (int j = 0; j < maxidx; j += 3) {
                    std::uint32_t const * face = model.Mesh.Indices.data() + j;
                    glm::vec3 const &     p1   = model.Mesh.Positions[face[0]];
                    glm::vec3 const &     p2   = model.Mesh.Positions[face[1]];
                    glm::vec3 const &     p3   = model.Mesh.Positions[face[2]];
                    if (! p_IntersectTriangle(its, ray, p1, p2, p3)) continue;
                    if (its.t < p_EPS1 || its.t > tmin) continue;
                    tmin = its.t, umin = its.u, vmin = its.v, modelIdx = i, meshIdx = j;
                }
            }
            if (tmin == 1e7) {
                result.IntersectState = false;
                return result;
            }
            auto const &          model     = InternalScene->Models[modelIdx];
            auto const &          normals   = model.Mesh.IsNormalAvailable() ? model.Mesh.Normals : model.Mesh.ComputeNormals();
            auto const &          texcoords = model.Mesh.IsTexCoordAvailable() ? model.Mesh.TexCoords : model.Mesh.GetEmptyTexCoords();
            std::uint32_t const * face      = model.Mesh.Indices.data() + meshIdx;
            glm::vec3 const &     p1        = model.Mesh.Positions[face[0]];
            glm::vec3 const &     p2        = model.Mesh.Positions[face[1]];
            glm::vec3 const &     p3        = model.Mesh.Positions[face[2]];
            glm::vec3 const &     n1        = normals[face[0]];
            glm::vec3 const &     n2        = normals[face[1]];
            glm::vec3 const &     n3        = normals[face[2]];
            glm::vec2 const &     uv1       = texcoords[face[0]];
            glm::vec2 const &     uv2       = texcoords[face[1]];
            glm::vec2 const &     uv3       = texcoords[face[2]];
            result.IntersectState           = true;
            auto const & material           = InternalScene->Materials[model.MaterialIndex];
            result.IntersectMode            = material.Blend;
            result.IntersectPosition        = (1.0f - umin - vmin) * p1 + umin * p2 + vmin * p3;
            result.IntersectNormal          = (1.0f - umin - vmin) * n1 + umin * n2 + vmin * n3;
            glm::vec2 uvCoord               = (1.0f - umin - vmin) * uv1 + umin * uv2 + vmin * uv3;
            result.IntersectAlbedo          = p_GetAlbedo(material, uvCoord);
            result.IntersectMetaSpec        = p_GetTexture(material.MetaSpec, uvCoord);

            return result;
        }
    };
    using p_RayIntersector = p_TrivialRayIntersector;



    class KDTree {

        // KD-Tree to accelerate find the nearest photons
        struct KDTree_KDNode {
            Photon *        photon_p; // Photon data
            KDTree_KDNode * left;     // Left childtree
            KDTree_KDNode * right;    // Right cildtree

            KDTree_KDNode(Photon & p):
                photon_p(&p), left(nullptr), right(nullptr) {}
        };
        // A compare function to compare photon by selected axis
        struct KDTree_PhotonCompareByAxis {
            int axis; // Axis to compare (0: x, 1: y, 2: z)
            KDTree_PhotonCompareByAxis(int _axis):
                axis(_axis) {}
            bool operator()(const Photon & photon_1, const Photon & photon_2) const {
                return photon_1.Position[axis] < photon_2.Position[axis];
            }
        };

        // DisNode:Store the distance to the query position
        struct KDTree_DisNode {
            float    distanceSquared; // squared distance to the query position
            Photon * photon_p;   // Pointer to the photon
            bool     operator<(const KDTree_DisNode & Disnode_) const {
                return distanceSquared < Disnode_.distanceSquared; // compare function for max_heap
            }
            KDTree_DisNode(float _distanceSquared, Photon * _photon_p):
                distanceSquared(_distanceSquared), photon_p(_photon_p) {}
        };

          // Alias for a priority queue of KDTree_DisNode (max-heap)
        using KDTree_NearestQueue = std::priority_queue<KDTree_DisNode>;


        std::vector<Photon> photons;        // Internal storage for photons
        KDTree_KDNode *     root = nullptr; // Root of the KD-Tree

        // build the KD-Tree
        void BuildTree(KDTree_KDNode *& node, int start, int end, int depth) {
            int axis = depth % 3;         // Choose the splitting axis (x, y, z)
            int mid  = (start + end) / 2; // Median index
            if (start >= end) return; // end of recursive

            if (! node) node = new KDTree_KDNode(photons[start]); // Create a new node if it doesn't exist

            if (end - start == 1) { // Leaf node: store the photon
                node->photon_p = &photons[start];
                node->left = nullptr;
                node->right    = nullptr;
                return;
            }

            // Sort photons along the current axis and find the median
            std::nth_element(photons.begin() + start, photons.begin() + mid, photons.begin() + end, KDTree_PhotonCompareByAxis(axis));

            node->photon_p = &photons[mid]; // Store the median photon

            // Recursively build left and right subtrees
            BuildTree(node->left, start, mid, depth + 1);//left treenode=>[start,mid]
            BuildTree(node->right, mid + 1, end, depth + 1);//right treenode=>[mid+1,right]
        }
        
        // Recursively find the nearest k photons and add them to the queue
        void FindNearestK_Photons(KDTree_KDNode * node, int start, int end, int depth, const glm::vec3 & queryPos, int k, KDTree_NearestQueue & queue, float maxDistanceSquared) const {
            if (start >= end) return; // end of recursive

            if (end - start == 1) {                                                                                         
                float distanceSquared = glm::dot(queryPos - node->photon_p->Position, queryPos - node->photon_p->Position); // Squared distance

                if (queue.size() < k) { // If the queue is not full, add the photon
                    if (maxDistanceSquared < 0 || distanceSquared <= maxDistanceSquared) {
                        queue.push(KDTree_DisNode(distanceSquared, node->photon_p));
                    }
                } else if (distanceSquared < queue.top().distanceSquared) {      // If the photon is closer than the farthest photon in the max_heap
                    queue.pop();                                                 // Remove the farthest photon
                    queue.push(KDTree_DisNode(distanceSquared, node->photon_p)); // Add the new photon
                }
                return;
            }

            int mid  = (start + end) / 2; // Median index
            int axis = depth % 3;     // Current splitting axis

            // select which tree to query depend on the axis
            if (queryPos[axis] <= node->photon_p->Position[axis]) {
                FindNearestK_Photons(node->left, start, mid, depth + 1, queryPos, k, queue, maxDistanceSquared);
            } else {
                FindNearestK_Photons(node->right, mid + 1, end, depth + 1, queryPos, k, queue, maxDistanceSquared);
            }

            // Check if the current photon should be added to the queue
            float distanceSquared = glm::dot(queryPos - node->photon_p->Position, queryPos - node->photon_p->Position); // Squared distance

            if (queue.size() < k) { // If the queue is not full, add the photon
                if (maxDistanceSquared < 0 || distanceSquared <= maxDistanceSquared) {
                    queue.push(KDTree_DisNode(distanceSquared, node->photon_p));
                }
            } else if (distanceSquared < queue.top().distanceSquared) {      // If the photon is closer than the farthest in the queue
                queue.pop();                                                 // Remove the farthest photon
                queue.push(KDTree_DisNode(distanceSquared, node->photon_p)); // Add the new photon
            }

            float currentMaxDistance  = queue.empty() ? maxDistanceSquared : queue.top().distanceSquared;                                      // Maximum allowed distance
            float axisDistanceSquared = (node->photon_p->Position[axis] - queryPos[axis]) * (node->photon_p->Position[axis] - queryPos[axis]); // Distance along the current axis

            if (axisDistanceSquared > maxDistanceSquared) return; // Skip if the other subtree is too far

            if (queue.size() < k || currentMaxDistance >= axisDistanceSquared) { // Search the other subtree if necessary
                if (queryPos[axis] > node->photon_p->Position[axis]) {
                    FindNearestK_Photons(node->left, start, mid, depth + 1, queryPos, k, queue, maxDistanceSquared);
                } else {
                    FindNearestK_Photons(node->right, mid + 1, end, depth + 1, queryPos, k, queue, maxDistanceSquared);
                }
            }
        }

        // Recursively clear the KD-Tree
        void ClearTree(KDTree_KDNode * node) {
            if (node) {
                ClearTree(node->left);  // Clear left subtree
                ClearTree(node->right); // Clear right subtree
                delete node;            // Delete the current node
            }
        }

    public:
        // Clear the KD-Tree
        void Clear() {
            ClearTree(root);
            root = nullptr;
        }

        // Destructor: clear the KD-Tree
        ~KDTree() {
            Clear();
        }

        // Build the KD-Tree from a list of photons
        void Build(const std::vector<Photon> & inputPhotons) {
            photons = inputPhotons;                // Copy photons to internal storage
            BuildTree(root, 0, photons.size(), 0); // Build the tree
        }

        // Find the nearest k photons to a given position
        void NearestKPhotons(const glm::vec3 & queryPos, int k, std::vector<Photon> & result, float maxDistance) const {
            KDTree_NearestQueue queue;                                                                 // Priority queue to store the nearest photons
            float maxDistanceSquared = maxDistance * maxDistance; // Squared max distance
            FindNearestK_Photons(root, 0, photons.size(), 0, queryPos, k, queue, maxDistanceSquared);   // Perform the search

            // Extract photons from the queue and store them in the result vector
            //int final_cnt = 0;
            while (! queue.empty()) {
                //final_cnt++;
                result.push_back(*queue.top().photon_p);
                queue.pop();
            }
            //printf("query_result_len:%d\n",final_cnt);
        }
    };

    class PhotonMapping {
    public:
        PhotonMapping() {}
        // Photon Tracing

        glm::vec3 generate_random_dir()
        {
            //glm::vec3 random_Dir = glm::normalize(glm::vec3(
            //    (rand() % 100) / 100.0f - 0.5f,
            //    (rand() % 100) / 100.0f - 0.5f,
            //    (rand() % 100) / 100.0f - 0.5f));


            static std::mt19937             rng(std::random_device {}()); // random_device
            std::normal_distribution<float> dist(0.0f, 1.0f);             // normal distribution

            glm::vec3 randomDir;
            do {
                randomDir = glm::vec3(
                    dist(rng), // x ~ N(0,1)
                    dist(rng), // y ~ N(0,1)
                    dist(rng)  // z ~ N(0,1)
                );
            } while (glm::length(randomDir) < 1e-5f); // avoid zero vector

            glm::vec3 random_Dir=glm::normalize(randomDir); // normalized

            return random_Dir;
        };
        glm::vec3 generate_random_hemidir(const glm::vec3& normal)
        {
            // generate a random light in the normal's hemi
            glm::vec3 tmp_random = generate_random_dir();
            float     tmp_dot    = glm::dot(tmp_random, normal);
            if ( tmp_dot< 0.0f)
            {
                tmp_random = tmp_random - 2.0f * tmp_dot * normal;
            }
            return tmp_random;
        }
        glm::vec3 generate_random_hemidir2(const glm::vec3 & normal) {
            // generate a random light in the normal's hemi
            glm::vec3 tmp_normal = -normal;
            glm::vec3 tmp_random = generate_random_dir();
            float     tmp_dot    = glm::dot(tmp_random, tmp_normal);
            if (tmp_dot < 0.0f) {
                tmp_random = tmp_random - 2.0f * tmp_dot * tmp_normal;
            }
            return tmp_random;
        }

        Photon Generate_point_photon(const Engine::Light & light,int total_photon_num) {
            float  _magic = 5.0;
            Photon ret_photon;
            ret_photon.Position           = light.Position;
            ret_photon.Power              = glm::vec3(_magic) * light.Intensity / glm::vec3(float(total_photon_num));
            glm::vec3 random_lightDir     = generate_random_dir();
            ret_photon.Direction          = random_lightDir;

            return ret_photon;
        };

        Photon BSDF_generate_photon(const Ray & ray, const p_RayHit& RayHit,const Photon& src_photon) {
            const glm::vec3 pos       = RayHit.IntersectPosition;
            const glm::vec3 n         = RayHit.IntersectNormal;
            const glm::vec3 kd        = RayHit.IntersectAlbedo;
            const glm::vec3 ks        = RayHit.IntersectMetaSpec;
            const float     alpha     = RayHit.IntersectAlbedo.w;
            const float     shininess = RayHit.IntersectMetaSpec.w * 256;

            Photon ret;
            glm::vec3 dir       = glm::normalize(generate_random_hemidir(n));//generate light depending on normal direction
            glm::vec3 h         = glm::normalize(-ray.Direction + dir);
            float     spec_coef = glm::pow(glm::max(glm::dot(h, n), 0.0f), shininess);
            float     diff_coef = glm::max(glm::dot(dir, n), 0.0f);
            ret.Power           = src_photon.Power * (diff_coef * kd + spec_coef * ks) * 8.0f;
            ret.Position        = pos;
            if (alpha < 0.9f || RayHit.IntersectMode == Engine::BlendMode::Transparent) {
                //printf("haha1\n");
                
                ret.Direction = glm::normalize(generate_random_hemidir2(n));
            }
            else {
                //printf("haha2\n");
              
                ret.Direction = dir;
            }

            return ret;
        };



        void Init_TracePhotons(const p_RayIntersector & intersector, int numPhotons, int maxBounces) {
            printf("begin_init\n");
            fflush(stdout);
            kd_tree->Clear();
            photons_vec.clear();
            //int push_cnt = 0;
            for (const Engine::Light & light : intersector.InternalScene->Lights) {
                // for each light, generate photons
                if (light.Type == Engine::LightType::Point) {
                    //printf("new light point \n");
                    // point light
                    for (int i = 0; i < numPhotons; ++i) {
                        Photon photon=Generate_point_photon(light,numPhotons); // generate photons

                        //int bounce_time=0;
                        //bool hit_flag = true;
                        for (int bounce = 0; bounce < maxBounces; ++bounce) {
                            Ray  ray(photon.Position, photon.Direction);
                            auto photonhit = intersector.IntersectRay(ray);

                            if (! photonhit.IntersectState) break; // photon miss

                            // save the photon if hits
                            photon.Position = photonhit.IntersectPosition;
                            photons_vec.push_back(photon);
                            //push_cnt++;
                            //if (hit_flag == true) bounce_time++;
                            //else hit_flag = true;


                            // randomly choose to continue
                            float diffuseProb = 0.7f;
                            if ((rand() % 100) / 100.0f < diffuseProb) break;

                            // update the photon

                            photon = BSDF_generate_photon(ray, photonhit,photon);
                            //photon.Power *= 0.5f;
                            //photon.Direction = glm::reflect(photon.Direction, photonhit.IntersectNormal);
                        }
                        //printf("bounce_time:%d\n", bounce_time);
                    }
                }
            }
            int photon_num_cnt = 0;
            for (const auto & photon : photons_vec) {
                photon_num_cnt++;
            }
            printf("photon_num_cnt:%d\n", photon_num_cnt);
            fflush(stdout);
            kd_tree= new KDTree;
            kd_tree->Build(photons_vec);
        }

        glm::vec3 calculate_photon_radiance(const p_RayHit & RayHit, Ray & ray, int near_photon_nums,float radius,bool enablekd) const
        {
            const glm::vec3 pos       = RayHit.IntersectPosition;
            const glm::vec3 n         = RayHit.IntersectNormal;
            const glm::vec3 kd        = RayHit.IntersectAlbedo;
            const glm::vec3 ks        = RayHit.IntersectMetaSpec;
            const float     alpha     = RayHit.IntersectAlbedo.w;
            const float     shininess = RayHit.IntersectMetaSpec.w * 256;

            glm::vec3 flux(0.0f);          
            float     maxDistance2 = 0.0f; 
            float     threshold_distance2 =radius*radius;
            if (enablekd)
            {
                //printf("enalble kd-tree~\n");
                float               cur_radius = radius;
                std::vector<Photon> near_k_photons;
                while (true) {
                    // NearestKPhotons(const glm::vec3 & queryPos, int k, std::vector<Photon> & result, float maxDistance)
                    kd_tree->NearestKPhotons(pos, near_photon_nums, near_k_photons, cur_radius);
                    if (near_k_photons.size() >= near_photon_nums) break; // get enough photons
                    near_k_photons.clear();
                    cur_radius *= 1.2f; // enlarge the size of search radius
                }

                // query photons
                for (const auto & photon : near_k_photons) {
                    float distance2 = glm::dot(photon.Position - pos, photon.Position - pos);
                    if (distance2 > threshold_distance2) continue;
                    maxDistance2    = glm::max(maxDistance2, distance2);

                    // Diffuse
                    glm::vec3 lightDir = glm::normalize(-photon.Direction);
                    float     diff_cos = glm::max(glm::dot(n, lightDir), 0.0f);
                    glm::vec3 diffuse  = kd * diff_cos * photon.Power;

                    // Blinn-Phong
                    glm::vec3 viewDir  = glm::normalize(-ray.Direction);
                    glm::vec3 halfDir  = glm::normalize(viewDir + lightDir);
                    float     spec_cos = glm::pow(glm::max(glm::dot(n, halfDir), 0.0f), shininess);
                    glm::vec3 specular = ks * spec_cos * photon.Power;

                    flux += (diffuse + specular);
                }
                if (maxDistance2 == 0.0f) return glm::vec3(0.0f);
                float area = glm::pi<float>() * maxDistance2;
                flux       = flux / area;

                return flux; 
            }
            else
            {
                // query photons
                int record_cnt = 0;
                for (const auto & photon :photons_vec) {
                    float distance2 = glm::dot(photon.Position - pos, photon.Position - pos);
                    if (distance2 > threshold_distance2) continue;
                    maxDistance2    = glm::max(maxDistance2, distance2);

                    // Diffuse
                    glm::vec3 lightDir = glm::normalize(-photon.Direction);
                    float     diff_cos = glm::max(glm::dot(n, lightDir), 0.0f);
                    glm::vec3 diffuse  = kd * diff_cos * photon.Power;

                    // Blinn-Phong
                    glm::vec3 viewDir  = glm::normalize(-ray.Direction);
                    glm::vec3 halfDir  = glm::normalize(viewDir + lightDir);
                    float     spec_cos = glm::pow(glm::max(glm::dot(n, halfDir), 0.0f), shininess);
                    glm::vec3 specular = ks * spec_cos * photon.Power;

                    flux += (diffuse + specular);
                    record_cnt++;
                    if (record_cnt == near_photon_nums) break;
                }

                if (maxDistance2 == 0.0f) return glm::vec3(0.0f);
                float area = glm::pi<float>() * maxDistance2;
                flux       = flux / area;

                return flux; 
            }
           
        }
        std::vector<Photon> photons_vec; // photons list
        KDTree*              kd_tree;// kd tree to accelerate photon query
    };
    bool      p_IntersectTriangle(p_Intersection & output, Ray const & ray, glm::vec3 const & p1, glm::vec3 const & p2, glm::vec3 const & p3);
    glm::vec3 RayTraceWithPhotonMapping(const p_RayIntersector & intersector, Ray ray, int maxDepth, const PhotonMapping & photonMapping,int k_near, float photonRadius,bool enablekd);
    } // namespace VCX::Labs::Rendering