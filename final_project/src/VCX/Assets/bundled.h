#pragma once

#include <array>
#include <string_view>

namespace VCX::Assets {
    inline constexpr auto DefaultIcons {
        std::to_array<std::string_view>({
            "assets/images/vcl-logo-32x32.png",
            "assets/images/vcl-logo-48x48.png",
        })
    };

    inline constexpr auto DefaultFonts {
        std::to_array<std::string_view>({
            "assets/fonts/Ubuntu.ttf",
            "assets/fonts/UbuntuMono.ttf",
        })
    };

    inline constexpr auto ExampleModels {
        std::to_array<std::string_view>({
            "assets/models/arma.obj",
            "assets/models/block.obj",
            "assets/models/cube.obj",
            "assets/models/dinosaur.obj",
            "assets/models/face.obj",
            "assets/models/fandisk.obj",
            "assets/models/rocker.obj",
            "assets/models/sphere.obj",
        })
    };

    enum class ExampleModel {
        Arma,
        Block,
        Cube,
        Dinosaur,
        Face,
        Fandisk,
        Rocker,
        Sphere,
    };

    inline constexpr auto ExampleScenes {
        std::to_array<std::string_view>({
            "assets/scenes/cornell_box/cornell_box.yaml",
            "assets/scenes/cornell_box_sphere_block/cornell_box_sphere_block.yaml",
            "assets/scenes/cornell_box_sphere_block_water/cornell_box_sphere_block_water.yaml"
        })
    };

    enum class ExampleScene {
        CornellBox,
        CornellBox_Sphere,
        CornellBox_Water,
    };
}
