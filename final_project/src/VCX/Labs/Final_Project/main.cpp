#include "Assets/bundled.h"
#include "Labs/Final_Project/App.h"

int main() {
    using namespace VCX;
    return Engine::RunApp<Labs::Rendering::App>(Engine::AppContextOptions {
        .Title      = "VCX Final Project: Phonton Mapping",
        .WindowSize = { 1024, 768 },
        .FontSize   = 16,

        .IconFileNames = Assets::DefaultIcons,
        .FontFileNames = Assets::DefaultFonts,
    });
}
