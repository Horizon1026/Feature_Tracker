#ifndef _OPTICAL_FLOW_LSSD_KLT_H_
#define _OPTICAL_FLOW_LSSD_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace FEATURE_TRACKER {

class OpticalFlowLssdKlt : public OpticalFlow {

public:
    OpticalFlowLssdKlt() : OpticalFlow() {}
    virtual ~OpticalFlowLssdKlt() = default;

    // Reference for member variables.
    float predict_theta() { return predict_theta_; }

    // Const reference for member variables.
    const float predict_theta() const { return predict_theta_; }

private:
    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) override;

    // Support for inverse/direct/fast method.
    void TrackOneFeature(const GrayImage &ref_image,
                         const GrayImage &cur_image,
                         const Vec2 &ref_pixel_uv,
                         Vec2 &cur_pixel_uv,
                         uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image,
                                         const GrayImage &cur_image,
                                         const Vec2 &ref_pixel_uv,
                                         const float &rotation,
                                         const Vec2 &translation,
                                         Mat3 &H,
                                         Vec3 &b);

    // Support for Sse method.

    // Support for Neon method.

private:
    // Rotation in se2 for prediction.
    float predict_theta_ = 0.0f;

};

}

#endif // end of _OPTICAL_FLOW_LSSD_KLT_H_
