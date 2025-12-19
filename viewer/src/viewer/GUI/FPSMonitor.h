#pragma once
#include <vector>
#include <viewer/Framework/RenderObject.h>

namespace viewer {

class FPSMonitor : public RenderObject
{
public:
    FPSMonitor()          = default;
    virtual ~FPSMonitor() = default;

    virtual void Draw(const CameraInfo& camera_info, const std::vector<LightInfo>& light_infos, const std::vector<ShadowMappingInfo>& shadow_mapping_infos) override;
    void         UpdateFPS(double elapsed_time);

protected:
    const size_t        m_history_size = 10;
    std::vector<double> m_time_history;
    double              m_avg_time = -1.0;
    double              m_fps      = 0.0;
};

}  // namespace viewer
