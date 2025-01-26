#include <unordered_map>
#include<vector>
#include<cmath>

#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>

#include "Labs/2-GeometryProcessing/DCEL.hpp"
#include "Labs/2-GeometryProcessing/tasks.h"


namespace VCX::Labs::GeometryProcessing {

#include "Labs/2-GeometryProcessing/marching_cubes_table.h"

    /******************* 1. Mesh Subdivision *****************/
    void SubdivisionMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations) {
        Engine::SurfaceMesh curr_mesh = input;
        // We do subdivison iteratively.
        for (std::uint32_t it = 0; it < numIterations; ++it) {
            // During each iteration, we first move curr_mesh into prev_mesh.
            Engine::SurfaceMesh prev_mesh;
            prev_mesh.Swap(curr_mesh);
            // Then we create doubly connected edge list.
            DCEL G(prev_mesh);
            if (! G.IsManifold()) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Non-manifold mesh.");
                return;
            }
            // Note that here curr_mesh has already been empty.
            // We reserve memory first for efficiency.
            curr_mesh.Positions.reserve(prev_mesh.Positions.size() * 3 / 2);
            curr_mesh.Indices.reserve(prev_mesh.Indices.size() * 4);
            // Then we iteratively update currently existing vertices.
            for (std::size_t i = 0; i < prev_mesh.Positions.size(); ++i) {
                // Update the currently existing vetex v from prev_mesh.Positions.
                // Then add the updated vertex into curr_mesh.Positions.
                auto v           = G.Vertex(i);
                auto neighbors   = v->Neighbors();
                // your code here:
                int vertex_degree = neighbors.size();
                float u_   = (vertex_degree == 3) ? (float)3 / 16 : (float)3 / (8 * vertex_degree);
                glm::vec3 new_v_pos     = (1-vertex_degree*u_)*prev_mesh.Positions[i];
                for (auto neighbor_i : neighbors)
                {
                    new_v_pos += prev_mesh.Positions[neighbor_i]*u_;
                }
                curr_mesh.Positions.push_back(new_v_pos);

            }
            // We create an array to store indices of the newly generated vertices.
            // Note: newIndices[i][j] is the index of vertex generated on the "opposite edge" of j-th
            //       vertex in the i-th triangle.
            std::vector<std::array<std::uint32_t, 3U>> newIndices(prev_mesh.Indices.size() / 3, { ~0U, ~0U, ~0U });
            // Iteratively process each halfedge.
            for (auto e : G.Edges()) {
                // newIndices[face index][vertex index] = index of the newly generated vertex
                newIndices[G.IndexOf(e->Face())][e->EdgeLabel()] = curr_mesh.Positions.size();
                auto eTwin                                       = e->TwinEdgeOr(nullptr);
                // eTwin stores the twin halfedge.
                if (! eTwin) {
                    // When there is no twin halfedge (so, e is a boundary edge):
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    int index1=e->To();
                    int index2 = e->From();

                    glm::vec3 pos1 = prev_mesh.Positions[index1];
                    glm::vec3 pos2 = prev_mesh.Positions[index2];

                    glm::vec3 new_pos = (pos1 + pos2) * (float)0.5;
                    curr_mesh.Positions.push_back(new_pos);
                } else {
                    // When the twin halfedge exists, we should also record:
                    //     newIndices[face index][vertex index] = index of the newly generated vertex
                    // Because G.Edges() will only traverse once for two halfedges,
                    //     we have to record twice.
                    newIndices[G.IndexOf(eTwin->Face())][e->TwinEdge()->EdgeLabel()] = curr_mesh.Positions.size();
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    int index0=e->To();
                    int index2 = e->From();
                    int index1 = e->OppositeVertex();
                    int index3 = e->TwinOppositeVertex();

                    glm::vec3 pos0 = prev_mesh.Positions[index0];
                    glm::vec3 pos2 = prev_mesh.Positions[index2];
                    glm::vec3 pos1 = prev_mesh.Positions[index1];
                    glm::vec3 pos3 = prev_mesh.Positions[index3];

                    glm::vec3 new_pos = (pos0 + pos2) * (float)3/(float)8 + (pos1 + pos3) * (float)1/(float)8;
                    curr_mesh.Positions.push_back(new_pos);
                    
                }
            }

            // Here we've already build all the vertices.     
            // Next, it's time to reconstruct face indices.
            for (std::size_t i = 0; i < prev_mesh.Indices.size(); i += 3U) {
                // For each face F in prev_mesh, we should create 4 sub-faces.
                // v0,v1,v2 are indices of vertices in F.
                // m0,m1,m2 are generated vertices on the edges of F.
                auto v0           = prev_mesh.Indices[i + 0U];
                auto v1           = prev_mesh.Indices[i + 1U];
                auto v2           = prev_mesh.Indices[i + 2U];
                auto [m0, m1, m2] = newIndices[i / 3U];
                // Note: m0 is on the opposite edge (v1-v2) to v0.
                // Please keep the correct indices order (consistent with order v0-v1-v2)
                //     when inserting new face indices.
                // toInsert[i][j] stores the j-th vertex index of the i-th sub-face.
                std::uint32_t toInsert[4][3] = {
                    // your code here:
                    {m1,v0,m2},
                    {m2,v1,m0},
                    {m0,v2,m1},
                    {m0,m1,m2}
                };
                // Do insertion.
                curr_mesh.Indices.insert(
                    curr_mesh.Indices.end(),
                    reinterpret_cast<std::uint32_t *>(toInsert),
                    reinterpret_cast<std::uint32_t *>(toInsert) + 12U
                );
            }

            if (curr_mesh.Positions.size() == 0) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Empty mesh.");
                output = input;
                return;
            }
        }
        // Update output.
        output.Swap(curr_mesh);
    }

    /******************* 2. Mesh Parameterization *****************/
    void Parameterization(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, const std::uint32_t numIterations) {
        // Copy.
        output = input;
        // Reset output.TexCoords.
        output.TexCoords.resize(input.Positions.size(), glm::vec2 { 0 });

        // Build DCEL.
        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::Parameterization(..): non-manifold mesh.");
            return;
        }

        // Set boundary UVs for boundary vertices.
        // your code here: directly edit output.TexCoords
        std::vector<int> bound_indices; //need order
        int bound_cnt = 0;
        int  last_bound_vertex_index=-1;
        for (std::size_t i = 0; i < input.Positions.size(); ++i) {
            // get vertex proxy and properties with index i
            DCEL::VertexProxy const * v   = G.Vertex(i);
            // do something with v & pos
            if (v->OnBoundary()) {
                bound_cnt++;
                last_bound_vertex_index = i;
            }
        }
        int tmp_cnt =1;
        bound_indices.push_back(last_bound_vertex_index);
        int old_find_cur = last_bound_vertex_index;
        int new_find_cur = G.Vertex(old_find_cur)->BoundaryNeighbors().first;

        while (new_find_cur!=last_bound_vertex_index)
        {
            bound_indices.push_back(new_find_cur);
            DCEL::VertexProxy const * v = G.Vertex(new_find_cur);
            if (v->BoundaryNeighbors().first !=old_find_cur)
            {
                old_find_cur=new_find_cur;
                new_find_cur = v->BoundaryNeighbors().first;
            }
            else
            {
                old_find_cur = new_find_cur;
                new_find_cur = v->BoundaryNeighbors().second;
            }
            tmp_cnt++;
        }

        double pi    = 4 * atan(1);
        float step_ = 2 * pi/bound_cnt;
        for (size_t i = 0; i < bound_cnt;i++) {
            // get vertex proxy and properties with index i
            size_t                    bound_index = bound_indices[i];
            DCEL::VertexProxy const * v   = G.Vertex(bound_index);
            glm::vec3                 pos = input.Positions[bound_index];
            // do something with v & pos
            if (v->OnBoundary())
            {
                float tmp_x         = 0.5 + 0.5 * cos(step_ * i);
                float tmp_y         = 0.5 + 0.5 * sin(step_ * i);
                output.TexCoords[bound_index] = glm::vec2 { tmp_x,tmp_y };
            }
        }

        // Solve equation via Gauss-Seidel Iterative Method.
        for (int k = 0; k < numIterations; ++k) {
            // your code here:
            for (std::size_t i = 0; i < input.Positions.size(); ++i) {
                // get vertex proxy and properties with index i
                size_t                    cnt = 0;
                glm::vec2                 tmp_tex = glm::vec2{ 0.0f, 0.0f };

                DCEL::VertexProxy const * v   = G.Vertex(i);
                // do something with v & pos
                if (!v->OnBoundary()) {
                    for (auto n_index : v->Neighbors())
                    {
                        //std::cout << cnt << std::endl;
                        cnt++;
                        glm::vec2 tex_ = output.TexCoords[n_index];
                        tmp_tex += tex_;
                    }
                    if (cnt != 0) {
                        glm::vec2 cnt_vec = glm::vec2 { cnt, cnt };
                        tmp_tex           = tmp_tex / cnt_vec;
                    }
                    output.TexCoords[i] = tmp_tex;
                }

            }
        }
    }

    /******************* 3. Mesh Simplification *****************/
    void SimplifyMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, float simplification_ratio) {

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-watertight mesh.");
            return;
        }

        // Copy.
        output = input;

        // Compute Kp matrix of the face f.
        auto UpdateQ {
            [&G, &output] (DCEL::Triangle const * f) -> glm::mat4 {
                glm::mat4 Kp=glm::mat4(0);
                // your code here:
                int point_index_0 = f->VertexIndex(0);
                int point_index_1 = f->VertexIndex(1);
                int point_index_2 = f->VertexIndex(2);

                glm::vec3 triangle_edge1 = output.Positions[point_index_1] - output.Positions[point_index_0];
                glm::vec3 triangle_edge2 = output.Positions[point_index_2] - output.Positions[point_index_1];
                glm::vec3 plane_norm     = glm::normalize(glm::cross(triangle_edge1, triangle_edge2));
                float tmp_d                 = -glm::dot(plane_norm, output.Positions[point_index_0]);
                glm::mat4 p_             = {
                    glm::vec4 {plane_norm, tmp_d},
                    glm::vec4 { 0, 0, 0, 0 },
                    glm::vec4 { 0, 0, 0, 0 },
                    glm::vec4 {0,0,0,0}
                };
                Kp = p_ * glm::transpose(p_);


                return Kp;
            }
        };

        // The struct to record contraction info.
        struct ContractionPair {
            DCEL::HalfEdge const * edge;            // which edge to contract; if $edge == nullptr$, it means this pair is no longer valid
            glm::vec4              targetPosition;  // the targetPosition $v$ for vertex $edge->From()$ to move to
            float                  cost;            // the cost $v.T * Qbar * v$
        };

        // Given an edge (v1->v2), the positions of its two endpoints (p1, p2) and the Q matrix (Q1+Q2),
        //     return the ContractionPair struct.
        static constexpr auto MakePair {
            [] (DCEL::HalfEdge const * edge,
                glm::vec3 const & p1,
                glm::vec3 const & p2,
                glm::mat4 const & Q
            ) -> ContractionPair {
                // your code here:
                ContractionPair ret_constraction_pair;
                glm::mat4       Q_diff = {
                    glm::vec4 { Q[0] },
                    glm::vec4 { Q[1] },
                    glm::vec4 { Q[2] },
                    glm::vec4 { 0, 0, 0, 1 }
                };
                Q_diff       = glm::transpose(Q_diff);// tanspose => diff matrix
                float uninverse_threshold = 0.001;
                glm::vec4 p_;
                float cost_;
                if (glm::determinant(Q_diff) < uninverse_threshold)
                {
                    p_ = glm::vec4({ p1 + p2 ,2}) / glm::vec4 { 2.0, 2.0, 2.0,2.0 };
                }
                else
                {
                    glm::mat4 tmp_inverse_matrix = glm::inverse(Q_diff);
                    p_        = { tmp_inverse_matrix[3][0],
                                 tmp_inverse_matrix[3][1],
                               tmp_inverse_matrix[3][2],
                                1.0 };
                }
                cost_ = glm::dot(p_, Q * p_);
                ret_constraction_pair.edge = edge;
                ret_constraction_pair.cost = cost_;
                ret_constraction_pair.targetPosition=p_;
                return ret_constraction_pair;
            }
        };

        // pair_map: map EdgeIdx to index of $pairs$
        // pairs:    store ContractionPair
        // Qv:       $Qv[idx]$ is the Q matrix of vertex with index $idx$
        // Kf:       $Kf[idx]$ is the Kp matrix of face with index $idx$
        std::unordered_map<DCEL::EdgeIdx, std::size_t> pair_map; 
        std::vector<ContractionPair>                  pairs; 
        std::vector<glm::mat4>                         Qv(G.NumOfVertices(), glm::mat4(0));
        std::vector<glm::mat4>                         Kf(G.NumOfFaces(),    glm::mat4(0));

        // Initially, we compute Q matrix for each faces and it accumulates at each vertex.
        for (auto f : G.Faces()) {
            auto Q                 = UpdateQ(f);
            Qv[f->VertexIndex(0)] += Q;
            Qv[f->VertexIndex(1)] += Q;
            Qv[f->VertexIndex(2)] += Q;
            Kf[G.IndexOf(f)]       = Q;
        }

        pair_map.reserve(G.NumOfFaces() * 3);
        pairs.reserve(G.NumOfFaces() * 3 / 2);

        // Initially, we make pairs from all the contractable edges.
        for (auto e : G.Edges()) {
            if (! G.IsContractable(e)) continue;
            auto v1                            = e->From();
            auto v2                            = e->To();
            auto pair                          = MakePair(e, input.Positions[v1], input.Positions[v2], Qv[v1] + Qv[v2]);
            pair_map[G.IndexOf(e)]             = pairs.size();
            pair_map[G.IndexOf(e->TwinEdge())] = pairs.size();
            pairs.emplace_back(pair);
        }

        // Loop until the number of vertices is less than $simplification_ratio * initial_size$.
        while (G.NumOfVertices() > simplification_ratio * Qv.size()) {
            // Find the contractable pair with minimal cost.
            std::size_t min_idx = ~0;
            for (std::size_t i = 1; i < pairs.size(); ++i) {
                if (! pairs[i].edge) continue;
                if (!~min_idx || pairs[i].cost < pairs[min_idx].cost) {
                    if (G.IsContractable(pairs[i].edge)) min_idx = i;
                    else pairs[i].edge = nullptr;
                }
            }
            if (!~min_idx) break;

            // top:    the contractable pair with minimal cost
            // v1:     the reserved vertex
            // v2:     the removed vertex
            // result: the contract result
            // ring:   the edge ring of vertex v1
            ContractionPair & top    = pairs[min_idx];
            auto               v1     = top.edge->From();
            auto               v2     = top.edge->To();
            auto               result = G.Contract(top.edge);
            auto               ring   = G.Vertex(v1)->Ring();

            top.edge             = nullptr;            // The contraction has already been done, so the pair is no longer valid. Mark it as invalid.
            output.Positions[v1] = top.targetPosition; // Update the positions.

            // We do something to repair $pair_map$ and $pairs$ because some edges and vertices no longer exist.
            for (int i = 0; i < 2; ++i) {
                DCEL::EdgeIdx removed           = G.IndexOf(result.removed_edges[i].first);
                DCEL::EdgeIdx collapsed         = G.IndexOf(result.collapsed_edges[i].second);
                pairs[pair_map[removed]].edge   = result.collapsed_edges[i].first;
                pairs[pair_map[collapsed]].edge = nullptr;
                pair_map[collapsed]             = pair_map[G.IndexOf(result.collapsed_edges[i].first)];
            }

            // For the two wing vertices, each of them lose one incident face.
            // So, we update the Q matrix.
            Qv[result.removed_faces[0].first] -= Kf[G.IndexOf(result.removed_faces[0].second)];
            Qv[result.removed_faces[1].first] -= Kf[G.IndexOf(result.removed_faces[1].second)];

            // For the vertex v1, Q matrix should be recomputed.
            // And as the position of v1 changed, all the vertices which are on the ring of v1 should update their Q matrix as well.
            Qv[v1] = glm::mat4(0);
            for (auto e : ring) {
                // your code here:
                //     1. Compute the new Kp matrix for $e->Face()$.
                //     2. According to the difference between the old Kp (in $Kf$) and the new Kp (computed in step 1),
                //        update Q matrix of each vertex on the ring (update $Qv$).
                //     3. Update Q matrix of vertex v1 as well (update $Qv$).
                //     4. Update $Kf$.


                glm::mat4 new_kp_e_face = UpdateQ(e->Face());
                for (int tmp_index = 0; tmp_index <= 2; tmp_index++)
                {
                    int target_index = e->Face()->VertexIndex(tmp_index);
                    if (target_index == v1) continue;
                    Qv[target_index] += (new_kp_e_face-Kf[G.IndexOf(e->Face())]);
                }

                Qv[v1] += new_kp_e_face;

                Kf[G.IndexOf(e->Face())] =new_kp_e_face;

            }

            // Finally, as the Q matrix changed, we should update the relative $ContractionPair$ in $pairs$.
            // Any pair with the Q matrix of its endpoints changed, should be remade by $MakePair$.
            // your code here:
            for (std::size_t i = 0; i < pairs.size(); ++i) {
                if (pairs[i].edge && (pairs[i].edge->From() == v1)) 
                {
                    pairs[i] = MakePair(pairs[i].edge, output.Positions[v1], output.Positions[pairs[i].edge->To()], Qv[v1] + Qv[pairs[i].edge->To()]);
                }
                else if (pairs[i].edge && (pairs[i].edge->To() == v1))
                {
                    pairs[i] = MakePair(pairs[i].edge, output.Positions[pairs[i].edge->From()], output.Positions[v1], Qv[v1] + Qv[pairs[i].edge->From()]);
                }
            }

        }

        // In the end, we check if the result mesh is watertight and manifold.
        if (! G.DebugWatertightManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Result is not watertight manifold.");
        }

        auto exported = G.ExportMesh();
        output.Indices.swap(exported.Indices);
    }

    /******************* 4. Mesh Smoothing *****************/
    void SmoothMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations, float lambda, bool useUniformWeight) {
        // Define function to compute cotangent value of the angle v1-vAngle-v2
        static constexpr auto GetCotangent {
            [] (glm::vec3 vAngle, glm::vec3 v1, glm::vec3 v2) -> float {
                // your code here:
                glm::vec3 edge1=v1-vAngle;
                glm::vec3 edge2=v2-vAngle;
                float     angle_cos = (glm::dot(glm::normalize(edge1),glm::normalize( edge2)));
                if (angle_cos <= 0) return 0.0f; //<0 ret 0
                else if (angle_cos == 1) return 10000.0f;
                float angle_cotangent = angle_cos / sqrt(1 - angle_cos * angle_cos);
             
                return angle_cotangent;
            }
        };

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-watertight mesh.");
            return;
        }

        Engine::SurfaceMesh prev_mesh;
        prev_mesh.Positions = input.Positions;
        for (std::uint32_t iter = 0; iter < numIterations; ++iter) {
            Engine::SurfaceMesh curr_mesh = prev_mesh;
            for (std::size_t i = 0; i < input.Positions.size(); ++i) {
                // your code here: curr_mesh.Positions[i] = ...
                glm::vec3 new_pos = { 0, 0, 0 };
                DCEL::VertexProxy const * v       = G.Vertex(i);
                glm::vec3  old_pos     = curr_mesh.Positions[i];
                // do something with v & pos
                float w_sum = 0;
                for (auto edge_ : v->Ring()) {
                    auto      twin_edge          = edge_->TwinEdge();
                    auto pointj_index = edge_->To();
                    auto      point1_index       = edge_->NextEdge()->To();
                    auto point2_index = twin_edge->NextEdge()->To();
                    glm::vec3 pj_pos       = curr_mesh.Positions[pointj_index];
                    glm::vec3 p1_pos       = curr_mesh.Positions[point1_index];
                    glm::vec3 p2_pos       = curr_mesh.Positions[point2_index];
                    float     get_cot_alpha      = GetCotangent(p1_pos, pj_pos, old_pos);
                    float     get_cot_beta    = GetCotangent(p2_pos, pj_pos, old_pos);
                    float     w_ij               = (useUniformWeight=true)?1:get_cot_alpha + get_cot_beta;
                    new_pos += w_ij * pj_pos;
                    w_sum += w_ij;
                }
                new_pos = new_pos / w_sum;
                new_pos = (1 - lambda) * old_pos + lambda * new_pos;
                curr_mesh.Positions[i] = new_pos;
            }
            // Move curr_mesh to prev_mesh.
            prev_mesh.Swap(curr_mesh);
        }
        // Move prev_mesh to output.
        output.Swap(prev_mesh);
        // Copy indices from input.
        output.Indices = input.Indices;
    }

    /******************* 5. Marching Cubes *****************/


    glm::vec3 interpolation_get_p_(const glm::vec3  p1, const glm::vec3  p2, float p1_sdf_val, float p2_sdf_val) {
        float sdf_val_21 = p2_sdf_val - p1_sdf_val;
        if (std::abs(p1_sdf_val) < 0.00001) return p1;
        if (std::abs(p2_sdf_val) < 0.00001) return p2;
        if (std::abs(sdf_val_21) < 0.00001) return p1;

        glm::vec3 tar_pos = (p2_sdf_val * p1 - p1_sdf_val * p2) / sdf_val_21;

        return tar_pos;
    }

    bool close_union_check(glm::vec3 point1, glm::vec3 point2,float threshold)
    {
        glm::vec3 point_distance = point1-point2;
  
        return (std::abs(point_distance.x) < threshold) && (std::abs(point_distance.y) < threshold) && (std::abs(point_distance.z) < threshold);
    }

    void MarchingCubes(Engine::SurfaceMesh & output, const std::function<float(const glm::vec3 &)> & sdf, const glm::vec3 & grid_min, const float dx, const int n) {
        // your code here:
        std::unordered_map < int, std::vector<std::pair<int,glm::vec3>>> cube_mesh_points_map;//cube_id = x*n*n+y*n+z to index & tar_point
        int       point_cnt=0;
        glm::vec3 unit_step[3] = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        
        for (int x_ = 0; x_ < n; x_++)
        {
            for (int y_ = 0; y_ < n; y_++)
            {
                for (int z_ = 0; z_ < n; z_++)
                {
                    glm::vec3 point_pos = unit_step[0] * float(x_)*dx + unit_step[1] * float(y_)*dx + unit_step[2] * float(z_)*dx + grid_min;
                    glm::vec3 points_positions[8];
                    uint32_t  v = 0;
                    for (int i = 0; i < 8; i++)
                    {
                        points_positions[i] = point_pos+glm::vec3 { (i & 1) * dx,((i >> 1) & 1) * dx,(i >> 2) * dx };
                    }
                    for (int i = 0; i < 8; i++)
                    {
                        //printf("%d %f\n", i, sdf(points_positions[i]));
                        if (sdf(points_positions[i]) > 0.0)
                        {
                            v += (1 << i);
                        }
                    }
                    //if(v!=255)printf("%d \n", v);
                    uint32_t edge_states = c_EdgeStateTable[v];
                    
                    if (edge_states == 0) continue;
                    std::unordered_map<int, std::pair<int,glm::vec3>> edge_point_map;
                    for (int j = 0; j < 12; j++)
                    {
                        if (((edge_states&(1<<j))==0))continue; // no mesh point
                        glm::vec3 edge_from = points_positions[0] + dx * (j & 1) * unit_step[((j >> 2) + 1) % 3] + dx * ((j >> 1) & 1) * unit_step[((j >> 2) + 2) % 3];
                        glm::vec3 edge_to   = edge_from + unit_step[j >> 2] * dx;
                        float     p1_sdf_val = sdf(edge_from);
                        float     p2_sdf_val = sdf(edge_to);
                        
                        glm::vec3 edge_tar_pos =interpolation_get_p_(edge_from,edge_to,p1_sdf_val,p2_sdf_val);
                        int       position_index = -1;
                        //bool      close_flag   = false;
                        //int       output_index_cnt = 0;
                        //for (auto tmp_pos : output.Positions)
                        //{
                        //    if (close_union_check(edge_tar_pos, tmp_pos, dx))
                        //    {
                        //        close_flag = true;
                        //        edge_tar_pos = tmp_pos;
                        //        break;
                        //    }
                        //    output_index_cnt++;
                        //}
                        //if(close_flag)
                        //{
                        //    edge_point_map[j] = std::make_pair(edge_tar_pos,output_index_cnt);
                        //}
                        //else
                        //{
                        //    edge_point_map[j] = std::make_pair(edge_tar_pos,-1);
                        //}

                        if (x_ > 0)
                        {
                            int nearby_cube = (x_ - 1) * n * n + y_ * n + z_;
                            for (auto nearby_edge_pair : cube_mesh_points_map[nearby_cube])
                            {
                                if (close_union_check(edge_tar_pos, nearby_edge_pair.second, dx/500))
                                {
                                    edge_tar_pos = nearby_edge_pair.second;
                                    position_index = nearby_edge_pair.first;
                                    break;
                                }
                            }
                        }
                        if (y_ > 0 && position_index == -1)
                        {
                            int nearby_cube = (x_ ) * n * n + (y_-1) * n + z_;
                            for (auto nearby_edge_pair : cube_mesh_points_map[nearby_cube]) {
                                if (close_union_check(edge_tar_pos, nearby_edge_pair.second, dx/500)) {
                                    edge_tar_pos   = nearby_edge_pair.second;
                                    position_index = nearby_edge_pair.first;
                                    break;
                                }
                            }
                        }
                        if (z_ > 0 && position_index == -1)
                        {
                            int nearby_cube = (x_) * n * n + y_ * n + z_-1;
                            for (auto nearby_edge_pair : cube_mesh_points_map[nearby_cube]) {
                                if (close_union_check(edge_tar_pos, nearby_edge_pair.second, dx/500)) {
                                    edge_tar_pos   = nearby_edge_pair.second;
                                    position_index = nearby_edge_pair.first;
                                    break;
                                }
                            }
                        }
                        edge_point_map[j] = std::make_pair(position_index, edge_tar_pos);
                        
                        //printf("%d ", j);
                        
                    }
                    //printf("\n");
                    int k = 0;
                    while (c_EdgeOrdsTable[v][3 * k] != -1)
                    {
                        int index[3] = { c_EdgeOrdsTable[v][3 * k],
                                         c_EdgeOrdsTable[v][3 * k+1],
                                         c_EdgeOrdsTable[v][3 * k+2] };
                        
                        for (int j = 2; j >=0; j--)
                        //for (int j = 0; j < 3; j++)
                        {
                            //printf("%d ", index[j]);
                            //printf("%d %f %f %f \n", point_cnt, edge_point_map[index[j]].x, edge_point_map[index[j]].y, edge_point_map[index[j]].z);
                            if (edge_point_map[index[j]].first != -1)//already in
                            {
                                cube_mesh_points_map[x_ * n * n + y_ * n + z_].push_back(std::make_pair(edge_point_map[index[j]].first, edge_point_map[index[j]].second));
                                output.Indices.push_back(edge_point_map[index[j]].first);
                            }
                            else
                            {
                                cube_mesh_points_map[x_ * n * n + y_ * n + z_].push_back(std::make_pair(point_cnt, edge_point_map[index[j]].second));
                                output.Indices.push_back(point_cnt);
                                output.Positions.push_back(edge_point_map[index[j]].second);
                                glm::vec3 tmp_pos        = edge_point_map[index[j]].second;
                                edge_point_map[index[j]] = std::make_pair(point_cnt,tmp_pos);
                                point_cnt++;
                            }

                        }
                        //printf("\n");
                        k++;
                    }
                    //printf("\n\n");
                }
            }
        }
    }
} // namespace VCX::Labs::GeometryProcessing
