#pragma once

#include <vector>

#include "Engine/app.h"
#include "Labs/Final_Project/CaseRayTracing.h"
#include "Labs/Final_Project/CasePhotonMapping.h"
#include "Labs/Common/UI.h"

namespace VCX::Labs::Rendering {
    class App : public Engine::IApp {
    private:
        Common::UI         _ui;

        CaseRayTracing     _caseRayTracing;
        CasePhotonMapping _casePhotonMapping;

        std::size_t        _caseId = 0;

        std::vector<std::reference_wrapper<Common::ICase>> _cases = {
            _caseRayTracing,
            _casePhotonMapping
        };

    public:
        App();

        void OnFrame() override;
    };
}
