#ifndef _OPTICAL_FLOW_BASIC_KLT_H_
#define _OPTICAL_FLOW_BASIC_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace feature_tracker {

class OpticalFlowBasicKlt: public OpticalFlow {

public:
    OpticalFlowBasicKlt(): OpticalFlow() {}
    virtual ~OpticalFlowBasicKlt() = default;

private:
    virtual bool TrackMultipleLevel(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) override;
    virtual bool TrackSingleLevel(const GrayImage &ref_image, const GrayImage &cur_image, const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) override;

    // Support for inverse and direct method.
    void TrackOneFeature(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Vec2 &cur_pixel_uv, uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Vec2 &cur_pixel_uv, Mat2 &H,
                                         Vec2 &b);

    // Support for fast method.
    void TrackOneFeatureFast(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Vec2 &cur_pixel_uv, uint8_t &status);
    void PrecomputeJacobianAndHessian(const std::vector<float> &ex_ref_patch, const std::vector<bool> &ex_ref_patch_pixel_valid, int32_t ex_ref_patch_rows,
                                      int32_t ex_ref_patch_cols, std::vector<float> &all_dx_in_ref_patch, std::vector<float> &all_dy_in_ref_patch,
                                      Mat2 &hessian);
    int32_t ComputeBias(const GrayImage &cur_image, const Vec2 &cur_pixel_uv, const std::vector<float> &ex_ref_patch,
                        const std::vector<bool> &ex_ref_patch_pixel_valid, int32_t ex_ref_patch_rows, int32_t ex_ref_patch_cols,
                        const std::vector<float> &all_dx_in_ref_patch, const std::vector<float> &all_dy_in_ref_patch, Vec2 &bias);

    // Support for Sse method.

    // Support for Neon method.
};

}  // namespace feature_tracker

#endif  // end of _OPTICAL_FLOW_BASIC_KLT_H_
