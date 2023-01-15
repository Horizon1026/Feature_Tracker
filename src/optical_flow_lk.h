#ifndef _OPTICAL_FLOW_LK_H_
#define _OPTICAL_FLOW_LK_H_

#include "optical_flow_datatype.h"
#include <eigen3/Eigen/Eigen>
#include <vector>

namespace OPTICAL_FLOW {

enum LkMethod : uint8_t {
    LK_INVERSE_LSE = 0,
    LK_DIRECT_LSE,
    LK_FAST,
};

struct LkOptions {
    uint32_t kMaxTrackPointsNumber = 200;
    uint32_t kMaxIteration = 10;
    int32_t kPatchRowHalfSize = 6;
    int32_t kPatchColHalfSize = 6;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 2.0f;
    LkMethod kMethod = LK_FAST;
};

class OpticalFlowLk {
public:
    explicit OpticalFlowLk() = default;
    virtual ~OpticalFlowLk() = default;

    bool TrackMultipleLevel(const ImagePyramid *ref_pyramid,
                            const ImagePyramid *cur_pyramid,
                            const std::vector<Vec2> &ref_points,
                            std::vector<Vec2> &cur_points,
                            std::vector<TrackStatus> &status);

    bool TrackSingleLevel(const Image *ref_image,
                          const Image *cur_image,
                          const std::vector<Vec2> &ref_points,
                          std::vector<Vec2> &cur_points,
                          std::vector<TrackStatus> &status);

    LkOptions &options() { return options_; }

private:
    void TrackOneFeatureFast(const Image *ref_image,
                             const Image *cur_image,
                             const Vec2 &ref_points,
                             Vec2 &cur_points,
                             TrackStatus &status);

    void TrackOneFeatureInverse(const Image *ref_image,
                                const Image *cur_image,
                                const Vec2 &ref_points,
                                Vec2 &cur_points,
                                TrackStatus &status);

    void TrackOneFeatureDirect(const Image *ref_image,
                               const Image *cur_image,
                               const Vec2 &ref_points,
                               Vec2 &cur_points,
                               TrackStatus &status);

    inline void GetPixelValueFromeBuffer(const Image *image,
                                         const int32_t row_idx_buf,
                                         const int32_t col_idx_buf,
                                         const float row_image,
                                         const float col_image,
                                         float *value) {
        float temp = pixel_values_in_patch_(row_idx_buf, col_idx_buf);

        if (temp > 0) {
            *value = temp;
        } else {
            *value = image->GetPixelValueNoCheck(row_image, col_image);
            pixel_values_in_patch_(row_idx_buf, col_idx_buf) = *value;
        }
    }

    void PrecomputeHessian(const Image *ref_image,
                           const Vec2 &ref_point,
                           Mat2 &H);

    float ComputeResidual(const Image *cur_image,
                          const Vec2 &cur_point,
                          Vec2 &b);

private:
    LkOptions options_;
    std::vector<Vec3> fx_fy_ti_;
    Mat pixel_values_in_patch_;
};

}

#endif
