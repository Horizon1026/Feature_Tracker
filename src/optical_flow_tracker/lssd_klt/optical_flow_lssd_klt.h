#ifndef _OPTICAL_FLOW_LSSD_KLT_H_
#define _OPTICAL_FLOW_LSSD_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace feature_tracker {

class OpticalFlowLssdKlt: public OpticalFlow {

public:
    OpticalFlowLssdKlt(): OpticalFlow() {}
    virtual ~OpticalFlowLssdKlt() = default;

    virtual std::string OpticalFlowMethodName() const override { return "Lssd-Klt"; }

    // Reference for member variables.
    Mat2 &predict_R_cr() { return predict_R_cr_; }
    bool &consider_patch_luminance() { return consider_patch_luminance_; }

    // Const reference for member variables.
    const Mat2 &predict_R_cr() const { return predict_R_cr_; }
    const bool &consider_patch_luminance() const { return consider_patch_luminance_; }

private:
    virtual bool TrackMultipleLevel(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) override;
    virtual bool TrackSingleLevel(const GrayImage &ref_image, const GrayImage &cur_image, const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) override;

    // Support for inverse/direct method.
    void TrackOneFeature(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Mat2 &R_cr, Vec2 &t_cr, uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr,
                                         Mat3 &hessian, Vec3 &bias);

    // Support for fast method.
    void TrackOneFeatureFast(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Mat2 &R_cr, Vec2 &t_cr, uint8_t &status);
    void PrecomputeJacobian(const std::vector<float> &ex_ref_patch, const std::vector<bool> &ex_ref_patch_pixel_valid, int32_t ex_ref_patch_rows,
                            int32_t ex_ref_patch_cols, std::vector<float> &all_dx_in_ref_patch, std::vector<float> &all_dy_in_ref_patch);
    uint32_t ExtractPatchInCurrentImage(const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr, int32_t cur_patch_rows,
                                        int32_t cur_patch_cols, std::vector<float> &cur_patch, std::vector<bool> &cur_patch_pixel_valid);
    int32_t ComputeHessianAndBias(const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr,
                                  const std::vector<float> &ex_ref_patch, const std::vector<bool> &ex_ref_patch_pixel_valid, int32_t ex_ref_patch_rows,
                                  int32_t ex_ref_patch_cols, const std::vector<float> &all_dx_in_ref_patch, const std::vector<float> &all_dy_in_ref_patch,
                                  const std::vector<float> &cur_patch, const std::vector<bool> &cur_patch_pixel_valid, Mat3 &hessian, Vec3 &bias);

    // Support for Sse method.

    // Support for Neon method.

private:
    // Support for prediction.
    Mat2 predict_R_cr_ = Mat2::Identity();
    bool consider_patch_luminance_ = false;
};

}  // namespace feature_tracker

#endif  // end of _OPTICAL_FLOW_LSSD_KLT_H_
