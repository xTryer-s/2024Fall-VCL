#pragma once

#include "Engine/GL/Frame.hpp"
#include "Engine/GL/Program.h"
#include "Labs/Final_Project/Content.h"
#include "Labs/Final_Project/SceneObject.h"
#include "Labs/Final_Project/photonMapping.h"
#include "Labs/Common/ICase.h"
#include "Labs/Common/ImageRGB.h"
#include "Labs/Common/OrbitCameraManager.h"

namespace VCX::Labs::Rendering {

    class CasePhotonMapping : public Common::ICase {
    public:
        CasePhotonMapping(std::initializer_list<Assets::ExampleScene> && scenes);
        ~CasePhotonMapping();

        virtual std::string_view const GetName() override { return "Photon Mapping"; }

        virtual void                     OnSetupPropsUI() override;
        virtual Common::CaseRenderResult OnRender(std::pair<std::uint32_t, std::uint32_t> const desiredSize) override;
        virtual void                     OnProcessInput(ImVec2 const & pos) override;

    private:
        std::vector<Assets::ExampleScene> const _scenes;
        Engine::GL::UniqueProgram               _program;
        Engine::GL::UniqueRenderFrame           _frame;
        SceneObject                             _sceneObject;
        Common::OrbitCameraManager              _cameraManager;

        Engine::GL::UniqueTexture2D _texture;
        p_RayIntersector              _intersector;
        PhotonMapping               _photonMapping;

        std::size_t                             _sceneIdx { 0 };
        bool                                    _enableZoom { true };
        bool                                    _enablekd { true };
        //bool                                    _enableShadow { true };
        int                                     _maximumDepth { 3 };
        int                                     _superSampleRate { 1 };
        int                                     _photon_nums { 2000 };
        int                                     _near_photon_nums { 100 };
        int                                     _photon_radius { 10 };
        int                                     _maximumPhotonBounce { 5 };
        std::size_t                             _pixelIndex { 0 };
        bool                                    _stopFlag { true };
        bool                                    _sceneDirty { true };
        bool                                    _treeDirty { true };
        bool                                    _resetDirty { true };
        Common::ImageRGB                        _buffer;
        bool                                    _resizable { true };

        std::thread _task;

        auto GetBufferSize() const { return std::pair(std::uint32_t(_buffer.GetSizeX()), std::uint32_t(_buffer.GetSizeY())); }

        char const *          GetSceneName(std::size_t const i) const { return Content::SceneNames[std::size_t(_scenes[i])].c_str(); }
        Engine::Scene const & GetScene(std::size_t const i) const { return Content::Scenes[std::size_t(_scenes[i])]; }
    };

} // namespace VCX::Labs::Rendering
