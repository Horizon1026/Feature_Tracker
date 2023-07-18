#ifndef _OPTICAL_FLOW_LK_H_
#define _OPTICAL_FLOW_LK_H_

#include "optical_flow.h"
#include <vector>

namespace FEATURE_TRACKER {

class OpticalFlowLk : public OpticalFlow {

public:
    OpticalFlowLk() : OpticalFlow() {}
    virtual ~OpticalFlowLk() = default;

private:
    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) override;
    virtual bool PrepareForTracking() override { return true; }

    // Support for kInverse and kDirect method.
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

    // Support for kFast method.
    void TrackOneFeatureFast(const GrayImage &ref_image,
                             const GrayImage &cur_image,
                             const Vec2 &ref_pixel_uv,
                             Vec2 &cur_pixel_uv,
                             uint8_t &status);
    uint32_t ExtractExtendPatchInReferenceImage(const GrayImage &ref_image,
                                                const Vec2 &ref_pixel_uv,
                                                int32_t ex_patch_rows,
                                                int32_t ex_patch_cols,
                                                std::vector<float> &ex_patch,
                                                std::vector<bool> &ex_patch_pixel_valid);
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
                        std::vector<float> &all_dx,
                        std::vector<float> &all_dy,
                        Vec2 &bias);

    // Support for Sse method.

    // Support for Neon method.
};

}

#endif
