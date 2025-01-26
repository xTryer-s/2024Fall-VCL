#include "Assets/bundled.h"
#include "Labs/Final_Project/App.h"

namespace VCX::Labs::Rendering {
    using namespace Assets;

    App::App() :
        _ui(Labs::Common::UIOptions { }),
        _caseRayTracing({ ExampleScene::CornellBox, ExampleScene::CornellBox_Sphere, ExampleScene::CornellBox_Water}),
        _casePhotonMapping({ ExampleScene::CornellBox, ExampleScene::CornellBox_Sphere, ExampleScene::CornellBox_Water}) {
    }

    void App::OnFrame() {
        _ui.Setup(_cases, _caseId);
    }
}
