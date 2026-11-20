#include "optical_flow_lssd_klt.h"

#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace feature_tracker {

void OpticalFlowLssdKlt::TrackOneFeatureFast(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Mat2 &R_cr, Vec2 &t_cr,
                                             uint8_t &status) {
    // Clear status firstly, then try to track.
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);

    // Confirm extended patch size. Extract it from reference image.
    ex_ref_patch().clear();
    ex_ref_patch_pixel_valid().clear();
    const uint32_t valid_pixel_num =
        ExtractExtendPatchInReferenceImage(ref_image, ref_pixel_uv, ex_ref_patch_rows(), ex_ref_patch_cols(), ex_ref_patch(), ex_ref_patch_pixel_valid());

    // If this feature has no valid pixel in patch, it can not be tracked.
    if (valid_pixel_num == 0) {
        status = static_cast<uint8_t>(TrackStatus::kOutside);
        return;
    }

    // Compute the image gradient of reference image.
    all_dx_in_ref_patch().clear();
    all_dy_in_ref_patch().clear();
    PrecomputeJacobian(ex_ref_patch(), ex_ref_patch_pixel_valid(), ex_ref_patch_rows(), ex_ref_patch_cols(), all_dx_in_ref_patch(), all_dy_in_ref_patch());

    // Compute incremental by iteration.
    Vec3 bias = Vec3::Zero();
    Mat3 hessian = Mat3::Zero();
    Mat2 previous_linear_step = Mat2::Zero();
    Vec2 previous_center_step = Vec2::Zero();
    uint32_t iter = 0;
    for (; iter < options().kMaxIteration; ++iter) {
        // Extract raw current intensities; normalization uses the common valid set below.
        cur_patch().clear();
        cur_patch_pixel_valid().clear();
        const uint32_t valid_pixel_num =
            ExtractPatchInCurrentImage(cur_image, ref_pixel_uv, R_cr, t_cr, patch_rows(), patch_cols(), cur_patch(), cur_patch_pixel_valid());
        if (valid_pixel_num == 0) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Compute hessian and bias.
        hessian.setZero();
        bias.setZero();
        BREAK_IF(ComputeHessianAndBias(cur_image, ref_pixel_uv, R_cr, t_cr, ex_ref_patch(), ex_ref_patch_pixel_valid(), ex_ref_patch_rows(),
                                       ex_ref_patch_cols(), all_dx_in_ref_patch(), all_dy_in_ref_patch(), cur_patch(), cur_patch_pixel_valid(), hessian,
                                       bias) == 0);

        // Validate observation.
        const float min_eig = Utility::MinEigenValue(hessian);
        if (min_eig < options().kMinEigenvalueThreshold) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Apply diagonal regularization for numerical stability.
        hessian.diagonal().array() += options().kHessianRegularizationLambda;

        // Solve incremental function using LDLT.
        const Eigen::LDLT<Mat3> ldlt_solver(hessian);
        if (ldlt_solver.info() != Eigen::Success) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }
        const Vec3 v = ldlt_solver.solve(bias);
        if (Eigen::isnan(v.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        const Mat2 old_rotation = R_cr;
        const Vec2 old_translation = t_cr;
        const Vec2 old_center = R_cr * ref_pixel_uv + t_cr;
        UpdateWarp(ref_pixel_uv, v, R_cr, t_cr);
        const Mat2 linear_step = R_cr - old_rotation;
        const Vec2 center_step = R_cr * ref_pixel_uv + t_cr - old_center;

        // Check if this step is converged.
        const float squared_step = v.squaredNorm();
        if (squared_step < options().kMaxSquaredConvergeStep) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
        if (iter > 0 && IsOscillatingWarpStep(linear_step, center_step, previous_linear_step, previous_center_step)) {
            // Reapply half of the last increment to keep the rotation on SO(2).
            R_cr = old_rotation;
            t_cr = old_translation;
            UpdateWarp(ref_pixel_uv, Vec3(0.5f * v), R_cr, t_cr);
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
        previous_linear_step = linear_step;
        previous_center_step = center_step;
    }
    // Exhausting the iteration budget is successful unless tracking exited early with a failure.
    if (iter == options().kMaxIteration) {
        status = static_cast<uint8_t>(TrackStatus::kTracked);
    }
}

void OpticalFlowLssdKlt::PrecomputeJacobian(const std::vector<float> &ex_ref_patch, const std::vector<uint8_t> &ex_ref_patch_pixel_valid,
                                            int32_t ex_ref_patch_rows, int32_t ex_ref_patch_cols, std::vector<float> &all_dx_in_ref_patch,
                                            std::vector<float> &all_dy_in_ref_patch) {
    const int32_t patch_rows = ex_ref_patch_rows - 2;
    const int32_t patch_cols = ex_ref_patch_cols - 2;

    for (int32_t row = 0; row < patch_rows; ++row) {
        for (int32_t col = 0; col < patch_cols; ++col) {
            const int32_t ex_index = (row + 1) * ex_ref_patch_cols + col + 1;
            const int32_t ex_index_left = ex_index - 1;
            const int32_t ex_index_right = ex_index + 1;
            const int32_t ex_index_top = ex_index - ex_ref_patch_cols;
            const int32_t ex_index_bottom = ex_index + ex_ref_patch_cols;

            if (ex_ref_patch_pixel_valid[ex_index_left] && ex_ref_patch_pixel_valid[ex_index_right] && ex_ref_patch_pixel_valid[ex_index_top] &&
                ex_ref_patch_pixel_valid[ex_index_bottom]) {
                // Compute dx and dy for jacobian.
                const float dx = 0.5f * (ex_ref_patch[ex_index_right] - ex_ref_patch[ex_index_left]);
                const float dy = 0.5f * (ex_ref_patch[ex_index_bottom] - ex_ref_patch[ex_index_top]);
                all_dx_in_ref_patch.emplace_back(dx);
                all_dy_in_ref_patch.emplace_back(dy);
            } else {
                all_dx_in_ref_patch.emplace_back(0.0f);
                all_dy_in_ref_patch.emplace_back(0.0f);
            }
        }
    }
}

uint32_t OpticalFlowLssdKlt::ExtractPatchInCurrentImage(const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr,
                                                        int32_t cur_patch_rows, int32_t cur_patch_cols, std::vector<float> &cur_patch,
                                                        std::vector<uint8_t> &cur_patch_pixel_valid) {
    // Bound the rotated rectangle: its axis-aligned half extent is abs(R) * patch_half_size.
    const Vec2 cur_pixel_uv = R_cr * ref_pixel_uv + t_cr;
    const Vec2 patch_half_size(cur_patch_cols / 2, cur_patch_rows / 2);
    const Vec2 half_extent = R_cr.cwiseAbs() * patch_half_size;
    const Vec2 min_pixel_uv = cur_pixel_uv - half_extent;
    const Vec2 max_pixel_uv = cur_pixel_uv + half_extent;
    const bool is_fully_inside = min_pixel_uv.allFinite() && max_pixel_uv.allFinite() && min_pixel_uv.x() >= 1.0f && min_pixel_uv.y() >= 1.0f &&
                                 max_pixel_uv.x() <= cur_image.cols() - 3.0f && max_pixel_uv.y() <= cur_image.rows() - 3.0f;

    float temp_value = 0.0f;
    if (!is_fully_inside) {
        // If this patch is partly outside of current image.
        uint32_t valid_pixel_cnt = 0;
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const Vec2 cur_patch_pixel_uv = R_cr * Vec2(col_i, row_i) + t_cr;
                const float row_j = cur_patch_pixel_uv.y();
                const float col_j = cur_patch_pixel_uv.x();

                if (cur_patch_pixel_uv.allFinite() && cur_image.GetPixelValue(row_j, col_j, &temp_value)) {
                    cur_patch.emplace_back(temp_value);
                    cur_patch_pixel_valid.emplace_back(1);
                    ++valid_pixel_cnt;
                } else {
                    cur_patch.emplace_back(0.0f);
                    cur_patch_pixel_valid.emplace_back(0);
                }
            }
        }

        return valid_pixel_cnt;
    } else {
        // If this patch is totally inside of current image.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const Vec2 cur_patch_pixel_uv = R_cr * Vec2(col_i, row_i) + t_cr;
                const float row_j = cur_patch_pixel_uv.y();
                const float col_j = cur_patch_pixel_uv.x();

                cur_patch.emplace_back(cur_image.GetPixelValueNoCheck(row_j, col_j));
                cur_patch_pixel_valid.emplace_back(1);
            }
        }
        return cur_patch.size();
    }
}

int32_t OpticalFlowLssdKlt::ComputeHessianAndBias(const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr,
                                                  const std::vector<float> &ex_ref_patch, const std::vector<uint8_t> &ex_ref_patch_pixel_valid,
                                                  int32_t ex_ref_patch_rows, int32_t ex_ref_patch_cols, const std::vector<float> &all_dx_in_ref_patch,
                                                  const std::vector<float> &all_dy_in_ref_patch, const std::vector<float> &cur_patch,
                                                  const std::vector<uint8_t> &cur_patch_pixel_valid, Mat3 &hessian, Vec3 &bias) {
    hessian.setZero();
    bias.setZero();
    int32_t num_of_valid_pixel = 0;
    float ref_average_value = 0.0f;
    float cur_average_value = 0.0f;
    Vec3 mean_intensity_jacobian = Vec3::Zero();
    const int32_t patch_cols = ex_ref_patch_cols - 2;

    // Both normalization and residuals use exactly the same pixels, including reference gradient support.
    const auto is_valid = [&](int32_t ex_index, int32_t index) {
        return cur_patch_pixel_valid[index] && ex_ref_patch_pixel_valid[ex_index] && ex_ref_patch_pixel_valid[ex_index - 1] &&
               ex_ref_patch_pixel_valid[ex_index + 1] && ex_ref_patch_pixel_valid[ex_index - ex_ref_patch_cols] &&
               ex_ref_patch_pixel_valid[ex_index + ex_ref_patch_cols];
    };
    for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
        for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
            const int32_t index = (drow + options().kPatchRowHalfSize) * patch_cols + dcol + options().kPatchColHalfSize;
            const int32_t ex_index = (drow + options().kPatchRowHalfSize + 1) * ex_ref_patch_cols + dcol + options().kPatchColHalfSize + 1;
            CONTINUE_IF(!is_valid(ex_index, index));
            const float dx = all_dx_in_ref_patch[index];
            const float dy = all_dy_in_ref_patch[index];
            mean_intensity_jacobian += Vec3(-drow * dx + dcol * dy, dx, dy);
            ref_average_value += ex_ref_patch[ex_index];
            cur_average_value += cur_patch[index];
            ++num_of_valid_pixel;
        }
    }
    if (num_of_valid_pixel == 0) {
        return 0;
    }
    ref_average_value /= static_cast<float>(num_of_valid_pixel);
    cur_average_value /= static_cast<float>(num_of_valid_pixel);
    mean_intensity_jacobian /= static_cast<float>(num_of_valid_pixel);
    if (ref_average_value <= options().kLuminanceNormalizationEpsilon) {
        mean_intensity_jacobian.setZero();
    }
    ref_average_value = std::max(ref_average_value, options().kLuminanceNormalizationEpsilon);
    cur_average_value = std::max(cur_average_value, options().kLuminanceNormalizationEpsilon);
    const float inv_ref_average = 1.0f / ref_average_value;
    const float inv_cur_average = 1.0f / cur_average_value;
    const Vec3 normalized_mean_jacobian = mean_intensity_jacobian * inv_ref_average;

    // Reference intensities and gradients remain raw: the common valid set may change each iteration.
    for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
        for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
            const int32_t index = (drow + options().kPatchRowHalfSize) * patch_cols + dcol + options().kPatchColHalfSize;
            const int32_t ex_index = (drow + options().kPatchRowHalfSize + 1) * ex_ref_patch_cols + dcol + options().kPatchColHalfSize + 1;
            CONTINUE_IF(!is_valid(ex_index, index));
            const float dx = all_dx_in_ref_patch[index];
            const float dy = all_dy_in_ref_patch[index];
            const float normalized_ref = ex_ref_patch[ex_index] * inv_ref_average;
            const float residual = cur_patch[index] * inv_cur_average - normalized_ref;
            const Vec3 jacobian = Vec3(-drow * dx + dcol * dy, dx, dy) * inv_ref_average - normalized_ref * normalized_mean_jacobian;
            hessian += jacobian * jacobian.transpose();
            bias += jacobian * residual;
        }
    }
    return num_of_valid_pixel;
}

}  // namespace feature_tracker
