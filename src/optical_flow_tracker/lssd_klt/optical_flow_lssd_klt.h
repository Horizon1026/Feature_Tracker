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
    Mat2 &predict_R_cr() { return predict_R_cr_; }

    // Const reference for member variables.
    const Mat2 &predict_R_cr() const { return predict_R_cr_; }

private:
    virtual bool TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                    const ImagePyramid &cur_pyramid,
                                    const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv,
                                    std::vector<uint8_t> &status) override;
    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) override;

    // Support for inverse/direct method.
    void TrackOneFeature(const GrayImage &ref_image,
                         const GrayImage &cur_image,
                         const Vec2 &ref_pixel_uv,
                         Mat2 &R_cr,
                         Vec2 &t_cr,
                         uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image,
                                         const GrayImage &cur_image,
                                         const Vec2 &ref_pixel_uv,
                                         const Mat2 &R_cr,
                                         const Vec2 &t_cr,
                                         Mat3 &H,
                                         Vec3 &b);

    // Support for fast method.
    void TrackOneFeatureFast(const GrayImage &ref_image,
                             const GrayImage &cur_image,
                             const Vec2 &ref_pixel_uv,
                             Mat2 &R_cr,
                             Vec2 &t_cr,
                             uint8_t &status);

    // Support for Sse method.

    // Support for Neon method.

private:
    // Support for prediction.
    Mat2 predict_R_cr_ = Mat2::Identity();

};

}

#endif // end of _OPTICAL_FLOW_LSSD_KLT_H_
