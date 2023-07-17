#include "optical_flow_lk.h"
#include "slam_memory.h"
#include "log_report.h"

#include "visualizor.h"

namespace FEATURE_TRACKER {

void OpticalFlowLk::TrackOneFeatureFast(const GrayImage &ref_image,
                                        const GrayImage &cur_image,
                                        const Vec2 &ref_pixel_uv,
                                        Vec2 &cur_pixel_uv,
                                        uint8_t &status) {
    // Confirm extended patch size. Extract it from reference image.
    const int32_t patch_rows = options().kPatchRowHalfSize * 2 + 1;
    const int32_t patch_cols = options().kPatchColHalfSize * 2 + 1;
    const int32_t ex_patch_rows = patch_rows + 2;
    const int32_t ex_patch_cols = patch_cols + 2;
    const int32_t ex_patch_size = ex_patch_rows * ex_patch_cols;
    std::vector<float> ex_patch;
    std::vector<bool> ex_patch_pixel_valid;
    ex_patch.reserve(ex_patch_size);
    ex_patch_pixel_valid.reserve(ex_patch_size);
    const uint32_t valid_pixel_num = ExtractExtendPatchInReferenceImage(ref_image, ref_pixel_uv, ex_patch_rows, ex_patch_cols, ex_patch, ex_patch_pixel_valid);

    // If this feature has no valid pixel in patch, it can not be tracked.
    if (!valid_pixel_num) {
        status = static_cast<uint8_t>(TrackStatus::kOutside);
        return;
    }

    // Precompute dx, dy, hessian matrix.
    const int32_t patch_size = patch_rows * patch_cols;
    std::vector<float> all_dx;
    std::vector<float> all_dy;
    all_dx.reserve(patch_size);
    all_dy.reserve(patch_size);
    Mat2 hessian = Mat2::Zero();
    PrecomputeJacobianAndHessian(ex_patch, ex_patch_pixel_valid, ex_patch_rows, ex_patch_cols, all_dx, all_dy, hessian);

    // Compute incremental by iteration.
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        Vec2 bias = Vec2::Zero();

        // Compute bias.
        // TODO:

        // Solve incremental function.
        const Vec2 v = hessian.ldlt().solve(bias);
        if (Eigen::isnan(v.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update cur_pixel_uv.
        cur_pixel_uv += v;

        // Check converge status.
        if (cur_pixel_uv.x() < 0 || cur_pixel_uv.x() > cur_image.cols() - 1 ||
            cur_pixel_uv.y() < 0 || cur_pixel_uv.y() > cur_image.rows() - 1) {
            status = static_cast<uint8_t>(TrackStatus::kOutside);
            break;
        }
        if (v.squaredNorm() < options().kMaxConvergeStep) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
    }
}

uint32_t OpticalFlowLk::ExtractExtendPatchInReferenceImage(const GrayImage &ref_image,
                                                           const Vec2 &ref_pixel_uv,
                                                           int32_t ex_patch_rows,
                                                           int32_t ex_patch_cols,
                                                           std::vector<float> &ex_patch,
                                                           std::vector<bool> &ex_patch_pixel_valid) {
    // Compute the weight for linear interpolar.
    const float int_pixel_row = std::floor(ref_pixel_uv.y());
    const float int_pixel_col = std::floor(ref_pixel_uv.x());
    const float dec_pixel_row = ref_pixel_uv.y() - int_pixel_row;
    const float dec_pixel_col = ref_pixel_uv.x() - int_pixel_col;
    const float w_top_left = (1.0f - dec_pixel_row) * (1.0f - dec_pixel_col);
    const float w_top_right = (1.0f - dec_pixel_row) * dec_pixel_col;
    const float w_bottom_left = dec_pixel_row * (1.0f - dec_pixel_col);
    const float w_bottom_right = dec_pixel_row * dec_pixel_col;

    // Extract patch from reference image.
    const int32_t min_ref_pixel_row = static_cast<int32_t>(int_pixel_row) - ex_patch_rows / 2;
    const int32_t min_ref_pixel_col = static_cast<int32_t>(int_pixel_col) - ex_patch_cols / 2;
    const int32_t max_ref_pixel_row = min_ref_pixel_row + ex_patch_rows;
    const int32_t max_ref_pixel_col = min_ref_pixel_col + ex_patch_cols;

    if (min_ref_pixel_row < 0 || max_ref_pixel_row > ref_image.rows() - 2 ||
        min_ref_pixel_col < 0 || max_ref_pixel_col > ref_image.cols() - 2) {
        // If this patch is partly outside of reference image.
        uint32_t valid_pixel_cnt = 0;
        for (int32_t row = min_ref_pixel_row; row < max_ref_pixel_row; ++row) {
            for (int32_t col = min_ref_pixel_col; col < max_ref_pixel_col; ++col) {
                if (row < 0 || row > ref_image.rows() - 2 || col < 0 || col > ref_image.cols()) {
                    ex_patch_pixel_valid.emplace_back(false);
                    ex_patch.emplace_back(0.0f);
                } else {
                    ex_patch_pixel_valid.emplace_back(true);
                    ex_patch.emplace_back(w_top_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col)) +
                                          w_top_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col + 1)) +
                                          w_bottom_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col)) +
                                          w_bottom_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col + 1)));
                    ++valid_pixel_cnt;
                }
            }
        }

        return valid_pixel_cnt;
    } else {
        // If this patch is totally inside of reference image.
        for (int32_t row = min_ref_pixel_row; row < max_ref_pixel_row; ++row) {
            for (int32_t col = min_ref_pixel_col; col < max_ref_pixel_col; ++col) {
                ex_patch_pixel_valid.emplace_back(true);
                ex_patch.emplace_back(w_top_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col)) +
                                      w_top_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col + 1)) +
                                      w_bottom_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col)) +
                                      w_bottom_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col + 1)));
            }
        }

        return ex_patch.size();
    }
}

void OpticalFlowLk::PrecomputeJacobianAndHessian(const std::vector<float> &ex_patch,
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
            const int32_t ex_index = row * ex_patch_cols + col + 1;
            const int32_t ex_index_left = ex_index - 1;
            const int32_t ex_index_right = ex_index + 1;
            const int32_t ex_index_top = ex_index + ex_patch_cols;
            const int32_t ex_index_bottom = ex_index - ex_patch_cols;

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

}
