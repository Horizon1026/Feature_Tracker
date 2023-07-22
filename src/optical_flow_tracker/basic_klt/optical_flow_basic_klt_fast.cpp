#include "optical_flow_basic_klt.h"
#include "slam_memory.h"
#include "log_report.h"
#include "slam_operations.h"

namespace FEATURE_TRACKER {

bool OpticalFlowBasicKlt::PrepareForTracking() {
    patch_rows_ = (options().kPatchRowHalfSize << 1) + 1;
    patch_cols_ = (options().kPatchColHalfSize << 1) + 1;
    patch_size_ = patch_rows_ * patch_cols_;

    ex_patch_rows_ = patch_rows_ + 2;
    ex_patch_cols_ = patch_cols_ + 2;
    ex_patch_size_ = ex_patch_rows_ * ex_patch_cols_;

    ex_patch_.reserve(ex_patch_size_);
    ex_patch_pixel_valid_.reserve(ex_patch_size_);

    all_dx_.reserve(patch_size_);
    all_dy_.reserve(patch_size_);

    return true;
}

void OpticalFlowBasicKlt::TrackOneFeatureFast(const GrayImage &ref_image,
                                              const GrayImage &cur_image,
                                              const Vec2 &ref_pixel_uv,
                                              Vec2 &cur_pixel_uv,
                                              uint8_t &status) {
    // Confirm extended patch size. Extract it from reference image.
    ex_patch_.clear();
    ex_patch_pixel_valid_.clear();
    const uint32_t valid_pixel_num = this->ExtractExtendPatchInReferenceImage(ref_image, ref_pixel_uv, ex_patch_rows_, ex_patch_cols_, ex_patch_, ex_patch_pixel_valid_);

    // If this feature has no valid pixel in patch, it can not be tracked.
    if (valid_pixel_num == 0) {
        status = static_cast<uint8_t>(TrackStatus::kOutside);
        return;
    }

    // Precompute dx, dy, hessian matrix.
    all_dx_.clear();
    all_dy_.clear();
    Mat2 hessian = Mat2::Zero();
    PrecomputeJacobianAndHessian(ex_patch_, ex_patch_pixel_valid_, ex_patch_rows_, ex_patch_cols_, all_dx_, all_dy_, hessian);

    // Compute incremental by iteration.
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);
    float last_squared_step = INFINITY;
    uint32_t large_step_cnt = 0;
    Vec2 bias = Vec2::Zero();
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Compute bias.
        BREAK_IF(ComputeBias(cur_image, cur_pixel_uv, ex_patch_, ex_patch_pixel_valid_,
            ex_patch_rows_, ex_patch_cols_, all_dx_, all_dy_, bias) == 0);

        // Solve incremental function.
        const Vec2 v = hessian.ldlt().solve(bias);
        if (Eigen::isnan(v.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update cur_pixel_uv.
        cur_pixel_uv += v;

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

    // Check if this feature is outside of current image.
    if (cur_pixel_uv.x() < 0 || cur_pixel_uv.x() > cur_image.cols() - 1 ||
        cur_pixel_uv.y() < 0 || cur_pixel_uv.y() > cur_image.rows() - 1) {
        status = static_cast<uint8_t>(TrackStatus::kOutside);
    }
}

void OpticalFlowBasicKlt::PrecomputeJacobianAndHessian(const std::vector<float> &ex_patch,
                                                       const std::vector<bool> &ex_patch_pixel_valid,
                                                       int32_t ex_patch_rows,
                                                       int32_t ex_patch_cols,
                                                       std::vector<float> &all_dx,
                                                       std::vector<float> &all_dy,
                                                       Mat2 &hessian) {
    const int32_t patch_rows = ex_patch_rows - 2;
    const int32_t patch_cols = ex_patch_cols - 2;
    hessian.setZero();

    for (int32_t row = 0; row < patch_rows; ++row) {
        for (int32_t col = 0; col < patch_cols; ++col) {
            const int32_t ex_index = (row + 1) * ex_patch_cols + col + 1;
            const int32_t ex_index_left = ex_index - 1;
            const int32_t ex_index_right = ex_index + 1;
            const int32_t ex_index_top = ex_index - ex_patch_cols;
            const int32_t ex_index_bottom = ex_index + ex_patch_cols;

            if (ex_patch_pixel_valid[ex_index_left] && ex_patch_pixel_valid[ex_index_right] &&
                ex_patch_pixel_valid[ex_index_top] && ex_patch_pixel_valid[ex_index_bottom]) {
                // Compute dx and dy for jacobian.
                const float dx = ex_patch[ex_index_right] - ex_patch[ex_index_left];
                const float dy = ex_patch[ex_index_bottom] - ex_patch[ex_index_top];
                all_dx.emplace_back(dx);
                all_dy.emplace_back(dy);

                // Compute hessian matrix.
                hessian(0, 0) += dx * dx;
                hessian(0, 1) += dx * dy;
                hessian(1, 1) += dy * dy;
            } else {
                all_dx.emplace_back(0.0f);
                all_dy.emplace_back(0.0f);
            }
        }
    }

    hessian(1, 0) = hessian(0, 1);
}

int32_t OpticalFlowBasicKlt::ComputeBias(const GrayImage &cur_image,
                                         const Vec2 &cur_pixel_uv,
                                         const std::vector<float> &ex_patch,
                                         const std::vector<bool> &ex_patch_pixel_valid,
                                         int32_t ex_patch_rows,
                                         int32_t ex_patch_cols,
                                         const std::vector<float> &all_dx,
                                         const std::vector<float> &all_dy,
                                         Vec2 &bias) {
    const int32_t patch_rows = ex_patch_rows - 2;
    const int32_t patch_cols = ex_patch_cols - 2;
    bias.setZero();

    // Compute the weight for linear interpolar.
    const float int_pixel_row = std::floor(cur_pixel_uv.y());
    const float int_pixel_col = std::floor(cur_pixel_uv.x());
    const float dec_pixel_row = cur_pixel_uv.y() - int_pixel_row;
    const float dec_pixel_col = cur_pixel_uv.x() - int_pixel_col;
    const float w_top_left = (1.0f - dec_pixel_row) * (1.0f - dec_pixel_col);
    const float w_top_right = (1.0f - dec_pixel_row) * dec_pixel_col;
    const float w_bottom_left = dec_pixel_row * (1.0f - dec_pixel_col);
    const float w_bottom_right = dec_pixel_row * dec_pixel_col;

    // Extract patch from current image, and compute bias.
    const int32_t min_cur_pixel_row = static_cast<int32_t>(int_pixel_row) - patch_rows / 2;
    const int32_t min_cur_pixel_col = static_cast<int32_t>(int_pixel_col) - patch_cols / 2;
    const int32_t max_cur_pixel_row = min_cur_pixel_row + patch_rows;
    const int32_t max_cur_pixel_col = min_cur_pixel_col + patch_cols;

    uint32_t valid_pixel_cnt = 0;
    if (min_cur_pixel_row < 0 || max_cur_pixel_row > cur_image.rows() - 2 ||
        min_cur_pixel_col < 0 || max_cur_pixel_col > cur_image.cols() - 2) {
        // If this patch is partly outside of reference image.
        for (int32_t row = min_cur_pixel_row; row < max_cur_pixel_row; ++row) {
            const int32_t row_in_ex_patch = row - min_cur_pixel_row + 1;
            const int32_t row_in_patch = row - min_cur_pixel_row;

            for (int32_t col = min_cur_pixel_col; col < max_cur_pixel_col; ++col) {
                CONTINUE_IF(row < 0 || row > cur_image.rows() - 2 || col < 0 || col > cur_image.cols() - 2);

                const int32_t col_in_ex_patch = col + 1 - min_cur_pixel_col;
                const int32_t index_in_ex_patch = row_in_ex_patch * ex_patch_cols + col_in_ex_patch;

                // If this pixel is invalid in ref or cur image, discard it.
                CONTINUE_IF(!ex_patch_pixel_valid[index_in_ex_patch]);

                // Compute pixel valud residual.
                const float ref_pixel_value = ex_patch[index_in_ex_patch];
                const float cur_pixel_value = w_top_left * static_cast<float>(cur_image.GetPixelValueNoCheck(row, col)) +
                                              w_top_right * static_cast<float>(cur_image.GetPixelValueNoCheck(row, col + 1)) +
                                              w_bottom_left * static_cast<float>(cur_image.GetPixelValueNoCheck(row + 1, col)) +
                                              w_bottom_right * static_cast<float>(cur_image.GetPixelValueNoCheck(row + 1, col + 1));
                const float dt = cur_pixel_value - ref_pixel_value;

                // Update bias.
                const int32_t &col_in_patch = col - min_cur_pixel_col;
                const int32_t index_in_patch = row_in_patch * patch_cols + col_in_patch;

                bias(0) -= all_dx[index_in_patch] * dt;
                bias(1) -= all_dy[index_in_patch] * dt;

                // Static valid pixel number.
                ++valid_pixel_cnt;
            }
        }

    } else {
        // If this patch is totally inside of reference image.
        for (int32_t row = min_cur_pixel_row; row < max_cur_pixel_row; ++row) {
            const int32_t row_in_ex_patch = row - min_cur_pixel_row + 1;
            const int32_t row_in_patch = row - min_cur_pixel_row;

            for (int32_t col = min_cur_pixel_col; col < max_cur_pixel_col; ++col) {
                const int32_t col_in_ex_patch = col + 1 - min_cur_pixel_col;
                const int32_t index_in_ex_patch = row_in_ex_patch * ex_patch_cols + col_in_ex_patch;

                // If this pixel is invalid in ref or cur image, discard it.
                CONTINUE_IF(!ex_patch_pixel_valid[index_in_ex_patch]);

                // Compute pixel valud residual.
                const float ref_pixel_value = ex_patch[index_in_ex_patch];
                const float cur_pixel_value = w_top_left * static_cast<float>(cur_image.GetPixelValueNoCheck(row, col)) +
                                              w_top_right * static_cast<float>(cur_image.GetPixelValueNoCheck(row, col + 1)) +
                                              w_bottom_left * static_cast<float>(cur_image.GetPixelValueNoCheck(row + 1, col)) +
                                              w_bottom_right * static_cast<float>(cur_image.GetPixelValueNoCheck(row + 1, col + 1));
                const float dt = cur_pixel_value - ref_pixel_value;

                // Update bias.
                const int32_t &col_in_patch = col - min_cur_pixel_col;
                const int32_t index_in_patch = row_in_patch * patch_cols + col_in_patch;

                bias(0) -= all_dx[index_in_patch] * dt;
                bias(1) -= all_dy[index_in_patch] * dt;

                // Static valid pixel number.
                ++valid_pixel_cnt;
            }
        }
    }

    return valid_pixel_cnt;
}

}
