#ifndef _OPTICAL_FLOW_LK_H_
#define _OPTICAL_FLOW_LK_H_

#include "optical_flow.h"
#include <vector>

namespace OPTICAL_FLOW {

class OpticalFlowLk : public OpticalFlow {

public:
    OpticalFlowLk() : OpticalFlow() {}
    virtual ~OpticalFlowLk() = default;

    virtual bool TrackSingleLevel(const Image &ref_image,
                                  const Image &cur_image,
                                  const std::vector<Vec2> &ref_points,
                                  std::vector<Vec2> &cur_points,
                                  std::vector<uint8_t> &status) override;

    virtual bool PrepareForTracking() override;

private:
    void TrackOneFeatureFast(const Image &ref_image,
                             const Image &cur_image,
                             const Vec2 &ref_points,
                             Vec2 &cur_points,
                             uint8_t &status);

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

    inline void GetPixelValueFromeBuffer(const Image &image,
                                         const int32_t row_idx_buf,
                                         const int32_t col_idx_buf,
                                         const float row_image,
                                         const float col_image,
                                         float *value) {
        float temp = pixel_values_in_patch_(row_idx_buf, col_idx_buf);

        if (temp > 0) {
            *value = temp;
        } else {
            *value = image.GetPixelValueNoCheck(row_image, col_image);
            pixel_values_in_patch_(row_idx_buf, col_idx_buf) = *value;
        }
    }

    void PrecomputeHessian(const Image &ref_image,
                           const Vec2 &ref_point,
                           Mat2 &H);

    float ComputeResidual(const Image &cur_image,
                          const Vec2 &cur_point,
                          Vec2 &b);

private:
    std::vector<Vec3> fx_fy_ti_;
    Mat pixel_values_in_patch_;
};

}

#endif
