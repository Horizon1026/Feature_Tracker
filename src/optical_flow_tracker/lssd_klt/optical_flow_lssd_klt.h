#ifndef _OPTICAL_FLOW_LSSD_KLT_H_
#define _OPTICAL_FLOW_LSSD_KLT_H_

#include "optical_flow.h"

#include "vector"

namespace feature_tracker {

class OpticalFlowLssdKlt: public OpticalFlow {

public:
    struct SubOptions {
        Mat2 kPredictionRcr = Mat2::Identity();
    };

public:
    OpticalFlowLssdKlt(): OpticalFlow() {}
    virtual ~OpticalFlowLssdKlt() = default;

    virtual std::string MethodName() const override { return "Lssd-Klt"; }

    // Reference for member variables.
    SubOptions &sub_options() { return sub_options_; }
    // Const reference for member variables.
    const SubOptions &sub_options() const { return sub_options_; }

private:
    virtual bool TrackMultipleLevel(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) override;
    virtual bool TrackSingleLevel(const GrayImage &ref_image, const GrayImage &cur_image, const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) override;

    // Apply a patch-centered increment, then recover the absolute SE(2) translation.
    void UpdateWarp(const Vec2 &ref_pixel_uv, const Vec3 &dx, Mat2 &R_cr, Vec2 &t_cr) const;

    // Support for inverse/direct method.
    void TrackOneFeature(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Mat2 &R_cr, Vec2 &t_cr, uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr,
                                         Mat3 &hessian, Vec3 &bias);

    // Support for fast method.
    void TrackOneFeatureFast(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Mat2 &R_cr, Vec2 &t_cr, uint8_t &status);
    void PrecomputeJacobian(const std::vector<float> &ex_ref_patch, const std::vector<uint8_t> &ex_ref_patch_pixel_valid, int32_t ex_ref_patch_rows,
                            int32_t ex_ref_patch_cols, std::vector<float> &all_dx_in_ref_patch, std::vector<float> &all_dy_in_ref_patch);
    uint32_t ExtractPatchInCurrentImage(const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr, int32_t cur_patch_rows,
                                        int32_t cur_patch_cols, std::vector<float> &cur_patch, std::vector<uint8_t> &cur_patch_pixel_valid);
    // Raw patches and gradients are normalized on their common valid set inside this function.
    int32_t ComputeHessianAndBias(const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr,
                                  const std::vector<float> &ex_ref_patch, const std::vector<uint8_t> &ex_ref_patch_pixel_valid, int32_t ex_ref_patch_rows,
                                  int32_t ex_ref_patch_cols, const std::vector<float> &all_dx_in_ref_patch, const std::vector<float> &all_dy_in_ref_patch,
                                  const std::vector<float> &cur_patch, const std::vector<uint8_t> &cur_patch_pixel_valid, Mat3 &hessian, Vec3 &bias);

    // Support for Sse method.

    // Support for Neon method.

private:
    SubOptions sub_options_;
};

}  // namespace feature_tracker

#endif  // end of _OPTICAL_FLOW_LSSD_KLT_H_
