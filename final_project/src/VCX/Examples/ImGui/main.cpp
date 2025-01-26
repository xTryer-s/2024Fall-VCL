#include <imgui.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>

#include "Assets/bundled.h"
#include "Engine/app.h"

int main() {
    class App : public VCX::Engine::IApp {
    public:
        App() {
            auto & style { ImGui::GetStyle() };
            style.ChildRounding     = 8;
            style.FrameRounding     = 8;
            style.GrabRounding      = 8;
            style.PopupRounding     = 8;
            style.ScrollbarRounding = 8;
            style.TabRounding       = 4;
            style.WindowRounding    = 8;
        }

        void OnFrame() override {
            ImGui::ShowDemoWindow();
        }
    };

    return VCX::Engine::RunApp<App>(VCX::Engine::AppContextOptions {
        .Title      = "VCX-Labs Example: ImGui",
        .WindowSize = { 800, 600 },
        .FontSize   = 16,

        .IconFileNames = VCX::Assets::DefaultIcons,
        .FontFileNames = VCX::Assets::DefaultFonts,
    });
}
