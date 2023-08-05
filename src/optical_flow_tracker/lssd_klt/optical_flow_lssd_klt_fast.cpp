#include "optical_flow_lssd_klt.h"
#include "slam_memory.h"
#include "log_report.h"
#include "slam_operations.h"

namespace FEATURE_TRACKER {

void OpticalFlowLssdKlt::TrackOneFeatureFast(const GrayImage &ref_image,
                                             const GrayImage &cur_image,
                                             const Vec2 &ref_pixel_uv,
                                             Mat2 &R_cr,
                                             Vec2 &t_cr,
                                             uint8_t &status) {
    // Confirm extended patch size. Extract it from reference image.
    ex_ref_patch().clear();
    ex_ref_patch_pixel_valid().clear();
    const uint32_t valid_pixel_num = ExtractExtendPatchInReferenceImage(ref_image, ref_pixel_uv, ex_ref_patch_rows(), ex_ref_patch_cols(), ex_ref_patch(), ex_ref_patch_pixel_valid());

    // If this feature has no valid pixel in patch, it can not be tracked.
    if (valid_pixel_num == 0) {
        status = static_cast<uint8_t>(TrackStatus::kOutside);
        return;
    }

    // Compute the average value.
    float ref_average_value = 0.0f;
    for (int32_t row = 1; row < ex_ref_patch_rows() - 1; ++row) {
        for (int32_t col = 1; col < ex_ref_patch_cols() - 1;++col) {
            ref_average_value += ex_ref_patch()[row * ex_ref_patch_cols() + col];
        }
    }
    ref_average_value /= static_cast<float>(valid_pixel_num);

    // Compute incremental by iteration.
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);
    float last_squared_step = INFINITY;
    uint32_t large_step_cnt = 0;
    Mat2 delta_R = Mat2::Identity();

    Vec3 bias = Vec3::Zero();
    Mat3 hessian = Mat3::Zero();
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Compute hessian and bias.
        // TODO:

        // Solve incremental function.
        const Vec3 v = hessian.ldlt().solve(bias);
        if (Eigen::isnan(v.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update rotation and translation.
        delta_R << 1, -v.x(), v.x(), 1;
        R_cr *= delta_R;
        R_cr /= R_cr.col(0).norm();
        t_cr += v.tail<2>();

        // Check if this step is converged.
        const float squared_step = v.squaredNorm();
        if (squared_step < last_squared_step) {
            last_squared_step = squared_step;
            large_step_cnt = 0;
        } else {
            ++large_step_cnt;
            BREAK_IF(large_step_cnt >= options().kMaxToleranceLargeStep);
        }
        if (squared_step < options().kMaxConvergeStep) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
    }
}

}
