#include <array>
#include <tuple>

#include <glm/glm.hpp>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>

#include "Assets/bundled.h"
#include "Engine/app.h"
#include "Engine/prelude.hpp"
#include "Engine/GL/Program.h"
#include "Engine/GL/RenderItem.h"
#include "Engine/GL/resource.hpp"

struct Vertex {
    glm::vec2 Position;
    glm::vec2 _0;
    glm::vec3 Color;
    glm::vec1 _1;
};

// note: this decl makes clang-format crash qaq
// clang-format off
static constexpr auto c_VertexData {
    std::to_array<Vertex>({
        { .Position { -0.5, -0.5 }, .Color { 1, 0, 0 }},
        { .Position {  0  ,  0.5 }, .Color { 0, 1, 0 }},
        { .Position {  0.5, -0.5 }, .Color { 0, 0, 1 }},
})};
// clang-format on
static constexpr auto c_ElementData {
    std::to_array<GLuint>({ 0, 1, 2 })
};

int main() {
    class App : public VCX::Engine::IApp {
    public:
        App():
            // clang-format off
            _program(
                VCX::Engine::GL::UniqueProgram({
                    VCX::Engine::GL::SharedShader("assets/shaders/triangle.vert"),
                    VCX::Engine::GL::SharedShader("assets/shaders/triangle.frag") })),
            // clang-format on
            _mesh(
                VCX::Engine::GL::VertexLayout()
                    .Add<Vertex>("vertex", VCX::Engine::GL::DrawFrequency::Static)
                    .At(0, &Vertex::Position)
                    .At(1, &Vertex::Color)) {
            _mesh.UpdateVertexBuffer("vertex", VCX::Engine::make_span_bytes<Vertex>(c_VertexData));
            _mesh.UpdateElementBuffer(c_ElementData);
        }

        void OnFrame() override {
            _mesh.Draw({ _program.Use() });
        }

    private:
        VCX::Engine::GL::UniqueProgram           _program;
        VCX::Engine::GL::UniqueIndexedRenderItem _mesh;
    };

    return VCX::Engine::RunApp<App>(VCX::Engine::AppContextOptions {
        .Title      = "VCX-Labs Example: Triangle",
        .WindowSize = { 800, 600 },
        .FontSize   = 16,

        .IconFileNames = VCX::Assets::DefaultIcons,
        .FontFileNames = VCX::Assets::DefaultFonts,
    });
}
