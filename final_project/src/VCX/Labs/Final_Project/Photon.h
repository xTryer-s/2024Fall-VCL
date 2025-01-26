#pragma once

#include <glm/glm.hpp>

namespace VCX::Labs::Rendering {

    struct Photon {
        glm::vec3 Position { 0, 0, 0 };
        glm::vec3 Direction { 0, 0, 0 };
        glm::vec3 Power { 1, 1, 1 };
        Photon() = default;
        Photon(const glm::vec3 & pos, const glm::vec3 & dir, const glm::vec3 &pow):
            Position { pos }, Direction { dir }, Power {pow} {}
    };

} // namespace VCX::Labs::Rendering
