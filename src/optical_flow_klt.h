#ifndef _OPTICAL_FLOW_KLT_H_
#define _OPTICAL_FLOW_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace OPTICAL_FLOW {

class OpticalFlowKlt : public OpticalFlow {

public:
    OpticalFlowKlt() : OpticalFlow() {}
    virtual ~OpticalFlowKlt() = default;

    virtual bool TrackSingleLevel(const Image &ref_image,
                                  const Image &cur_image,
                                  const std::vector<Vec2> &ref_points,
                                  std::vector<Vec2> &cur_points,
                                  std::vector<uint8_t> &status) override;

    virtual bool PrepareForTracking() override;

private:
    void TrackOneFeatureInverse(const Image &ref_image,
                                const Image &cur_image,
                                const Vec2 &ref_points,
                                Vec2 &cur_points,
                                uint8_t &status);

    void TrackOneFeatureDirect(const Image &ref_image,
                               const Image &cur_image,
                               const Vec2 &ref_points,
                               Vec2 &cur_points,
                               uint8_t &status);

private:
    std::vector<Vec3> fx_fy_ti_;
};

}

#endif
