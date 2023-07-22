#ifndef _OPTICAL_FLOW_BASIC_KLT_H_
#define _OPTICAL_FLOW_BASIC_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace FEATURE_TRACKER {

class OpticalFlowBasicKlt : public OpticalFlow {

public:
    OpticalFlowBasicKlt() : OpticalFlow() {}
    virtual ~OpticalFlowBasicKlt() = default;

private:
    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) override;
    virtual bool PrepareForTracking() override;

    // Support for inverse and direct method.
    void TrackOneFeature(const GrayImage &ref_image,
                         const GrayImage &cur_image,
                         const Vec2 &ref_pixel_uv,
                         Vec2 &cur_pixel_uv,
                         uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image,
                                         const GrayImage &cur_image,
                                         const Vec2 &ref_pixel_uv,
                                         const Vec2 &cur_pixel_uv,
                                         Mat2 &H,
                                         Vec2 &b);

    // Support for fast method.
    void TrackOneFeatureFast(const GrayImage &ref_image,
                             const GrayImage &cur_image,
                             const Vec2 &ref_pixel_uv,
                             Vec2 &cur_pixel_uv,
                             uint8_t &status);
    void PrecomputeJacobianAndHessian(const std::vector<float> &ex_patch,
                                      const std::vector<bool> &ex_patch_pixel_valid,
                                      int32_t ex_patch_rows,
                                      int32_t ex_patch_cols,
                                      std::vector<float> &all_dx,
                                      std::vector<float> &all_dy,
                                      Mat2 &hessian);
    int32_t ComputeBias(const GrayImage &cur_image,
                        const Vec2 &cur_pixel_uv,
                        const std::vector<float> &ex_patch,
                        const std::vector<bool> &ex_patch_pixel_valid,
                        int32_t ex_patch_rows,
                        int32_t ex_patch_cols,
                        const std::vector<float> &all_dx,
                        const std::vector<float> &all_dy,
                        Vec2 &bias);

    // Support for Sse method.

    // Support for Neon method.

private:
    // Variable support for fast method.
    std::vector<float> ex_patch_;
    std::vector<bool> ex_patch_pixel_valid_;
    std::vector<float> all_dx_;
    std::vector<float> all_dy_;

    int32_t patch_rows_ = 0;
    int32_t patch_cols_ = 0;
    int32_t patch_size_ = 0;
    int32_t ex_patch_rows_ = 0;
    int32_t ex_patch_cols_ = 0;
    int32_t ex_patch_size_ = 0;

};

}

#endif // end of _OPTICAL_FLOW_BASIC_KLT_H_
