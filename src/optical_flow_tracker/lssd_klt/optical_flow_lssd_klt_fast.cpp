#include "optical_flow_lssd_klt.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace FEATURE_TRACKER {

void OpticalFlowLssdKlt::TrackOneFeatureFast(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Mat2 &R_cr, Vec2 &t_cr,
                                             uint8_t &status) {
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

    // Compute the average value for reference patch.
    if (consider_patch_luminance_) {
        float ref_average_value = 0.0f;
        for (int32_t row = 1; row < ex_ref_patch_rows() - 1; ++row) {
            for (int32_t col = 1; col < ex_ref_patch_cols() - 1; ++col) {
                ref_average_value += ex_ref_patch()[row * ex_ref_patch_cols() + col];
            }
        }
        ref_average_value /= static_cast<float>(valid_pixel_num);

        // Scale dx, dy and pixel value in reference patch.
        for (auto &dx: all_dx_in_ref_patch()) {
            dx /= ref_average_value;
        }
        for (auto &dy: all_dy_in_ref_patch()) {
            dy /= ref_average_value;
        }
        for (auto &value: ex_ref_patch()) {
            value /= ref_average_value;
        }
    }

    // Compute incremental by iteration.
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);
    float last_squared_step = INFINITY;
    uint32_t large_step_cnt = 0;
    Mat2 delta_R = Mat2::Identity();

    Vec3 bias = Vec3::Zero();
    Mat3 hessian = Mat3::Zero();
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Extract patch in current image, and compute average value.
        cur_patch().clear();
        cur_patch_pixel_valid().clear();
        const uint32_t valid_pixel_num =
            ExtractPatchInCurrentImage(cur_image, ref_pixel_uv, R_cr, t_cr, patch_rows(), patch_cols(), cur_patch(), cur_patch_pixel_valid());
        BREAK_IF(valid_pixel_num == 0);

        // Compute the average value for reference patch.
        if (consider_patch_luminance_) {
            float cur_average_value = 0.0f;
            for (int32_t row = 1; row < patch_rows() - 1; ++row) {
                for (int32_t col = 1; col < patch_cols() - 1; ++col) {
                    cur_average_value += cur_patch()[row * patch_cols() + col];
                }
            }
            cur_average_value /= static_cast<float>(valid_pixel_num);

            // Scale pixel value in current patch.
            for (auto &value: cur_patch()) {
                value /= cur_average_value;
            }
        }

        // Compute hessian and bias.
        hessian.setZero();
        bias.setZero();
        BREAK_IF(ComputeHessianAndBias(cur_image, ref_pixel_uv, R_cr, t_cr, ex_ref_patch(), ex_ref_patch_pixel_valid(), ex_ref_patch_rows(),
                                       ex_ref_patch_cols(), all_dx_in_ref_patch(), all_dy_in_ref_patch(), cur_patch(), cur_patch_pixel_valid(), hessian,
                                       bias) == 0);

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

void OpticalFlowLssdKlt::PrecomputeJacobian(const std::vector<float> &ex_ref_patch, const std::vector<bool> &ex_ref_patch_pixel_valid,
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
                const float dx = ex_ref_patch[ex_index_right] - ex_ref_patch[ex_index_left];
                const float dy = ex_ref_patch[ex_index_bottom] - ex_ref_patch[ex_index_top];
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
                                                        std::vector<bool> &cur_patch_pixel_valid) {
    // Check if this patch inside of current image.
    const Vec2 cur_pixel_uv = R_cr * ref_pixel_uv + t_cr;
    const int32_t min_cur_pixel_row = static_cast<int32_t>(cur_pixel_uv.y()) - cur_patch_rows;
    const int32_t min_cur_pixel_col = static_cast<int32_t>(cur_pixel_uv.x()) - cur_patch_cols;
    const int32_t max_cur_pixel_row = min_cur_pixel_row + cur_patch_rows * 2;
    const int32_t max_cur_pixel_col = min_cur_pixel_col + cur_patch_cols * 2;

    float temp_value = 0.0f;
    if (min_cur_pixel_row < 0 || max_cur_pixel_row > cur_image.rows() - 2 || min_cur_pixel_col < 0 || max_cur_pixel_col > cur_image.cols() - 2) {
        // If this patch is partly outside of current image.
        uint32_t valid_pixel_cnt = 0;
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const Vec2 cur_patch_pixel_uv = R_cr * Vec2(col_i, row_i) + t_cr;
                const float row_j = cur_patch_pixel_uv.y();
                const float col_j = cur_patch_pixel_uv.x();

                if (cur_image.GetPixelValue(row_j, col_j, &temp_value)) {
                    cur_patch.emplace_back(temp_value);
                    cur_patch_pixel_valid.emplace_back(true);
                    ++valid_pixel_cnt;
                } else {
                    cur_patch.emplace_back(0.0f);
                    cur_patch_pixel_valid.emplace_back(false);
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
                cur_patch_pixel_valid.emplace_back(true);
            }
        }
        return cur_patch.size();
    }
}

int32_t OpticalFlowLssdKlt::ComputeHessianAndBias(const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr, const Vec2 &t_cr,
                                                  const std::vector<float> &ex_ref_patch, const std::vector<bool> &ex_ref_patch_pixel_valid,
                                                  int32_t ex_ref_patch_rows, int32_t ex_ref_patch_cols, const std::vector<float> &all_dx_in_ref_patch,
                                                  const std::vector<float> &all_dy_in_ref_patch, const std::vector<float> &cur_patch,
                                                  const std::vector<bool> &cur_patch_pixel_valid, Mat3 &hessian, Vec3 &bias) {
    int32_t num_of_valid_pixel = 0;
    Mat1x3 jacobian = Mat1x3::Zero();

    // For inverse optical flow, use reference image to compute gradient.
    for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
        for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
            // Localize the pixel in ex_ref_patch and cur_patch. They have different rows and cols.
            const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
            const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
            const int32_t ex_index = (drow + options().kPatchRowHalfSize + 1) * ex_ref_patch_cols + dcol + options().kPatchColHalfSize + 1;
            const int32_t index = (drow + options().kPatchRowHalfSize) * (ex_ref_patch_cols - 2) + dcol + options().kPatchColHalfSize;

            // If the pixel is both valid in ex_ref_patch and cur_patch.
            if (ex_ref_patch_pixel_valid[ex_index] && cur_patch_pixel_valid[index]) {
                jacobian << Vec2(all_dx_in_ref_patch[index], all_dy_in_ref_patch[index]).dot(R_cr * Vec2(-row_i, col_i)), all_dx_in_ref_patch[index],
                    all_dy_in_ref_patch[index];
                const float residual = cur_patch[index] - ex_ref_patch[ex_index];

                hessian += jacobian.transpose() * jacobian;
                bias -= jacobian.transpose() * residual;

                ++num_of_valid_pixel;
            }
        }
    }

    return num_of_valid_pixel;
}

}  // namespace FEATURE_TRACKER
