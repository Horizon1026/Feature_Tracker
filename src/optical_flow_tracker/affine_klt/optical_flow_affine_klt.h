#ifndef _OPTICAL_FLOW_AFFINE_KLT_H_
#define _OPTICAL_FLOW_AFFINE_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace FEATURE_TRACKER {

class OpticalFlowAffineKlt : public OpticalFlow {

public:
    OpticalFlowAffineKlt() : OpticalFlow() {}
    virtual ~OpticalFlowAffineKlt() = default;

    // Reference for member variables.
    Mat2 &predict_affine() { return predict_affine_; }

    // Const reference for member variables.
    const Mat2 &predict_affine() const { return predict_affine_; }

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

    // Support for inverse and direct method.
    void TrackOneFeature(const GrayImage &ref_image,
                         const GrayImage &cur_image,
                         const Vec2 &ref_pixel_uv,
                         Vec2 &cur_pixel_uv,
                         Mat2 &affine,
                         uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image,
                                         const GrayImage &cur_image,
                                         const Vec2 &ref_pixel_uv,
                                         const Vec2 &cur_pixel_uv,
                                         const Mat2 &affine,
                                         Mat6 &hessian,
                                         Vec6 &bias);

    // Support for fast method.
    void TrackOneFeatureFast(const GrayImage &ref_image,
                             const GrayImage &cur_image,
                             const Vec2 &ref_pixel_uv,
                             Vec2 &cur_pixel_uv,
                             Mat2 &affine,
                             uint8_t &status);
    void PrecomputeJacobianAndHessian(const std::vector<float> &ex_ref_patch,
                                      const std::vector<bool> &ex_ref_patch_pixel_valid,
                                      int32_t ex_ref_patch_rows,
                                      int32_t ex_ref_patch_cols,
                                      const Vec2 &cur_pixel_uv,
                                      std::vector<float> &all_dx_in_ref_patch,
                                      std::vector<float> &all_dy_in_ref_patch,
                                      Mat6 &hessian);
    int32_t ComputeBias(const GrayImage &cur_image,
                        const Vec2 &cur_pixel_uv,
                        const std::vector<float> &ex_ref_patch,
                        const std::vector<bool> &ex_ref_patch_pixel_valid,
                        int32_t ex_ref_patch_rows,
                        int32_t ex_ref_patch_cols,
                        const std::vector<float> &all_dx_in_ref_patch,
                        const std::vector<float> &all_dy_in_ref_patch,
                        const Mat2 &affine,
                        Vec6 &bias);

    // Support for Sse method.

    // Support for Neon method.

private:
    // Support for prediction.
    Mat2 predict_affine_ = Mat2::Identity();

};

}

#endif // end of _OPTICAL_FLOW_AFFINE_KLT_H_
