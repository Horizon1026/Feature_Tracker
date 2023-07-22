#include "optical_flow_affine_klt.h"
#include "slam_operations.h"
#include "log_report.h"

namespace FEATURE_TRACKER {

namespace {
    static Vec3 kInfinityVec3 = Vec3(INFINITY, INFINITY, INFINITY);
}

bool OpticalFlowAffineKlt::PrepareForTracking() {
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

void OpticalFlowAffineKlt::TrackOneFeatureFast(const GrayImage &ref_image,
                                               const GrayImage &cur_image,
                                               const Vec2 &ref_point,
                                               Vec2 &cur_point,
                                               uint8_t &status) {
    // Mat6 H = Mat6::Zero();
    // Vec6 b = Vec6::Zero();
    // Mat2 A = Mat2::Identity();    /* Affine trasform matrix. */

    // // Precompute H, fx, fy and ti.
    // fx_fy_ti_.clear();
    // float temp_value[6] = { 0 };

    // for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
    //     for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
    //         float row_i = static_cast<float>(drow) + ref_point.y();
    //         float col_i = static_cast<float>(dcol) + ref_point.x();
    //         float row_j = static_cast<float>(drow) + cur_point.y();
    //         float col_j = static_cast<float>(dcol) + cur_point.x();

    //         // Compute pixel gradient
    //         if (ref_image.GetPixelValue(row_i, col_i - 1.0f, temp_value) &&
    //             ref_image.GetPixelValue(row_i, col_i + 1.0f, temp_value + 1) &&
    //             ref_image.GetPixelValue(row_i - 1.0f, col_i, temp_value + 2) &&
    //             ref_image.GetPixelValue(row_i + 1.0f, col_i, temp_value + 3) &&
    //             ref_image.GetPixelValue(row_i, col_i, temp_value + 4)) {
    //             fx_fy_ti_.emplace_back(Vec3(temp_value[1] - temp_value[0],
    //                                         temp_value[3] - temp_value[2],
    //                                         temp_value[4]));

    //             const float &fx = fx_fy_ti_.back().x();
    //             const float &fy = fx_fy_ti_.back().y();
    //             const float &x = col_j;
    //             const float &y = row_j;

    //             const float xx = x * x;
    //             const float yy = y * y;
    //             const float xy = x * y;
    //             const float fxfx = fx * fx;
    //             const float fyfy = fy * fy;
    //             const float fxfy = fx * fy;

    //             H(0, 0) += xx * fxfx;
    //             H(0, 1) += xx * fxfy;
    //             H(0, 2) += xy * fxfx;
    //             H(0, 3) += xy * fxfy;
    //             H(0, 4) += x * fxfx;
    //             H(0, 5) += x * fxfy;
    //             H(1, 1) += xx * fyfy;
    //             H(1, 3) += xy * fyfy;
    //             H(1, 5) += x * fyfy;
    //             H(2, 2) += yy * fxfx;
    //             H(2, 3) += yy * fxfy;
    //             H(2, 4) += y * fxfx;
    //             H(2, 5) += y * fxfy;
    //             H(3, 3) += yy * fyfy;
    //             H(3, 5) += y * fyfy;
    //             H(4, 4) += fxfx;
    //             H(4, 5) += fxfy;
    //             H(5, 5) += fyfy;
    //         } else {
    //             fx_fy_ti_.emplace_back(kInfinityVec3);
    //         }
    //     }
    // }

    // H(1, 2) = H(0, 3);
    // H(1, 4) = H(0, 5);
    // H(3, 4) = H(2, 3);
    // for (uint32_t i = 0; i < 6; ++i) {
    //     for (uint32_t j = i; j < 6; ++j) {
    //         if (i != j) {
    //             H(j, i) = H(i, j);
    //         }
    //     }
    // }

    // // Iterate to compute optical flow.
    // for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
    //     b.setZero();
    //     int32_t num_of_valid_pixel = 0;

    //     // Compute each pixel in the patch, create H * v = b
    //     uint32_t idx = 0;
    //     for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
    //         for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
    //             Vec2 affined_dcol_drow = A * Vec2(dcol, drow);
    //             float row_j = affined_dcol_drow.y() + cur_point.y();
    //             float col_j = affined_dcol_drow.x() + cur_point.x();

    //             // Compute pixel gradient
    //             if (cur_image.GetPixelValue(row_j, col_j, temp_value + 5) &&
    //                 !std::isinf(fx_fy_ti_[idx].x())) {
    //                 const float fx = fx_fy_ti_[idx].x();
    //                 const float fy = fx_fy_ti_[idx].y();
    //                 const float ft = temp_value[5] - fx_fy_ti_[idx].z();

    //                 float &x = col_j;
    //                 float &y = row_j;

    //                 b(0) -= ft * x * fx;
    //                 b(1) -= ft * x * fy;
    //                 b(2) -= ft * y * fx;
    //                 b(3) -= ft * y * fy;
    //                 b(4) -= ft * fx;
    //                 b(5) -= ft * fy;

    //                 ++num_of_valid_pixel;
    //             }

    //             ++idx;
    //         }
    //     }

    //     // Solve H * z = b, update cur_pixel_uv.
    //     Vec6 z = H.ldlt().solve(b);
    //     Vec2 v = z.head<2>() * cur_point.x() + z.segment<2>(2) * cur_point.y() + z.tail<2>();

    //     if (std::isnan(v(0)) || std::isnan(v(1))) {
    //         status = static_cast<uint8_t>(TrackStatus::kNumericError);
    //         break;
    //     }

    //     cur_point.x() += v(0);
    //     cur_point.y() += v(1);

    //     // Update affine translation matrix.
    //     A.col(0) += z.head<2>();
    //     A.col(1) += z.segment<2>(2);

    //     if (cur_point.x() < 0 || cur_point.x() > cur_image.cols() ||
    //         cur_point.y() < 0 || cur_point.y() > cur_image.rows()) {
    //         status = static_cast<uint8_t>(TrackStatus::kOutside);
    //         break;
    //     }

    //     if (v.squaredNorm() < options().kMaxConvergeStep) {
    //         status = static_cast<uint8_t>(TrackStatus::kTracked);
    //         break;
    //     }
    // }
}

void OpticalFlowAffineKlt::PrecomputeJacobianAndHessian(const std::vector<float> &ex_patch,
                                                        const std::vector<bool> &ex_patch_pixel_valid,
                                                        int32_t ex_patch_rows,
                                                        int32_t ex_patch_cols,
                                                        std::vector<float> &all_dx,
                                                        std::vector<float> &all_dy,
                                                        Mat6 &hessian) {
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

int32_t OpticalFlowAffineKlt::ComputeBias(const GrayImage &cur_image,
                                          const Vec2 &cur_pixel_uv,
                                          const std::vector<float> &ex_patch,
                                          const std::vector<bool> &ex_patch_pixel_valid,
                                          int32_t ex_patch_rows,
                                          int32_t ex_patch_cols,
                                          std::vector<float> &all_dx,
                                          std::vector<float> &all_dy,
                                          Vec6 &bias) {
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
    const int32_t min_ref_pixel_row = static_cast<int32_t>(int_pixel_row) - patch_rows / 2;
    const int32_t min_ref_pixel_col = static_cast<int32_t>(int_pixel_col) - patch_cols / 2;
    const int32_t max_ref_pixel_row = min_ref_pixel_row + patch_rows;
    const int32_t max_ref_pixel_col = min_ref_pixel_col + patch_cols;

    uint32_t valid_pixel_cnt = 0;
    if (min_ref_pixel_row < 0 || max_ref_pixel_row > cur_image.rows() - 2 ||
        min_ref_pixel_col < 0 || max_ref_pixel_col > cur_image.cols() - 2) {
        // If this patch is partly outside of reference image.
        for (int32_t row = min_ref_pixel_row; row < max_ref_pixel_row; ++row) {
            const int32_t row_in_ex_patch = row - min_ref_pixel_row + 1;
            const int32_t row_in_patch = row - min_ref_pixel_row;

            for (int32_t col = min_ref_pixel_col; col < max_ref_pixel_col; ++col) {
                CONTINUE_IF(row < 0 || row > cur_image.rows() - 2 || col < 0 || col > cur_image.cols() - 2);

                const int32_t col_in_ex_patch = col + 1 - min_ref_pixel_col;
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
                const int32_t &col_in_patch = col - min_ref_pixel_col;
                const int32_t index_in_patch = row_in_patch * patch_cols + col_in_patch;

                bias(0) -= all_dx[index_in_patch] * dt;
                bias(1) -= all_dy[index_in_patch] * dt;

                // Static valid pixel number.
                ++valid_pixel_cnt;
            }
        }

    } else {
        // If this patch is totally inside of reference image.
        for (int32_t row = min_ref_pixel_row; row < max_ref_pixel_row; ++row) {
            const int32_t row_in_ex_patch = row - min_ref_pixel_row + 1;
            const int32_t row_in_patch = row - min_ref_pixel_row;

            for (int32_t col = min_ref_pixel_col; col < max_ref_pixel_col; ++col) {
                const int32_t col_in_ex_patch = col + 1 - min_ref_pixel_col;
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
                const int32_t &col_in_patch = col - min_ref_pixel_col;
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
