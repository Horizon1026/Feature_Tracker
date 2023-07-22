#ifndef _OPTICAL_FLOW_AFFINE_KLT_H_
#define _OPTICAL_FLOW_AFFINE_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace FEATURE_TRACKER {

class OpticalFlowAffineKlt : public OpticalFlow {

public:
    OpticalFlowAffineKlt() : OpticalFlow() {}
    virtual ~OpticalFlowAffineKlt() = default;

    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) override;

    virtual bool PrepareForTracking() override;

private:
    void TrackOneFeature(const GrayImage &ref_image,
                         const GrayImage &cur_image,
                         const Vec2 &ref_pixel_uv,
                         Vec2 &cur_pixel_uv,
                         uint8_t &status);

    int32_t ConstructIncrementalFunction(const GrayImage &ref_image,
                                         const GrayImage &cur_image,
                                         const Vec2 &ref_point,
                                         const Vec2 &cur_point,
                                         Mat2 &A,
                                         Mat6 &H,
                                         Vec6 &b);

    void TrackOneFeatureFast(const GrayImage &ref_image,
                             const GrayImage &cur_image,
                             const Vec2 &ref_pixel_uv,
                             Vec2 &cur_pixel_uv,
                             uint8_t &status);

private:
    std::vector<Vec3> fx_fy_ti_;
};

}

#endif // end of _OPTICAL_FLOW_AFFINE_KLT_H_
