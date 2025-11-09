#include "optical_flow_affine_klt.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace feature_tracker {

void OpticalFlowAffineKlt::TrackOneFeatureFast(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Vec2 &cur_pixel_uv,
                                               Mat2 &affine, uint8_t &status) {
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

    // Precompute dx, dy, hessian matrix.
    all_dx_in_ref_patch().clear();
    all_dy_in_ref_patch().clear();
    Mat6 hessian = Mat6::Zero();
    PrecomputeJacobianAndHessian(ex_ref_patch(), ex_ref_patch_pixel_valid(), ex_ref_patch_rows(), ex_ref_patch_cols(), cur_pixel_uv, all_dx_in_ref_patch(),
                                 all_dy_in_ref_patch(), hessian);

    // Compute incremental by iteration.
    Vec6 bias = Vec6::Zero();
    float last_squared_step = INFINITY;
    uint32_t large_step_cnt = 0;
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);

    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {

        // Compute bias.
        BREAK_IF(ComputeBias(cur_image, cur_pixel_uv, ex_ref_patch(), ex_ref_patch_pixel_valid(), ex_ref_patch_rows(), ex_ref_patch_cols(),
                             all_dx_in_ref_patch(), all_dy_in_ref_patch(), affine, bias) == 0);

        // Solve incremental function.
        const Vec6 z = hessian.ldlt().solve(bias);
        if (Eigen::isnan(z.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update cur_pixel_uv.
        const Vec2 v = z.head<2>() * cur_pixel_uv.x() + z.segment<2>(2) * cur_pixel_uv.y() + z.tail<2>();
        cur_pixel_uv += v;

        // Update affine transform matrix.
        affine.col(0) += z.head<2>();
        affine.col(1) += z.segment<2>(2);

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

void OpticalFlowAffineKlt::PrecomputeJacobianAndHessian(const std::vector<float> &ex_ref_patch, const std::vector<bool> &ex_ref_patch_pixel_valid,
                                                        int32_t ex_ref_patch_rows, int32_t ex_ref_patch_cols, const Vec2 &cur_pixel_uv,
                                                        std::vector<float> &all_dx_in_ref_patch, std::vector<float> &all_dy_in_ref_patch, Mat6 &hessian) {
    const int32_t patch_rows = ex_ref_patch_rows - 2;
    const int32_t patch_cols = ex_ref_patch_cols - 2;
    hessian.setZero();

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

                // Precompute temp value.
                const float x = static_cast<float>(col - options().kPatchColHalfSize) + cur_pixel_uv.x();
                const float y = static_cast<float>(row - options().kPatchRowHalfSize) + cur_pixel_uv.y();
                const float xx = x * x;
                const float yy = y * y;
                const float xy = x * y;
                const float dxdx = dx * dx;
                const float dydy = dy * dy;
                const float dxdy = dx * dy;

                // Compute hessian matrix.
                hessian(0, 0) += xx * dxdx;
                hessian(0, 1) += xx * dxdy;
                hessian(0, 2) += xy * dxdx;
                hessian(0, 3) += xy * dxdy;
                hessian(0, 4) += x * dxdx;
                hessian(0, 5) += x * dxdy;
                hessian(1, 1) += xx * dydy;
                hessian(1, 3) += xy * dydy;
                hessian(1, 5) += x * dydy;
                hessian(2, 2) += yy * dxdx;
                hessian(2, 3) += yy * dxdy;
                hessian(2, 4) += y * dxdx;
                hessian(2, 5) += y * dxdy;
                hessian(3, 3) += yy * dydy;
                hessian(3, 5) += y * dydy;
                hessian(4, 4) += dxdx;
                hessian(4, 5) += dxdy;
                hessian(5, 5) += dydy;
            } else {
                all_dx_in_ref_patch.emplace_back(0.0f);
                all_dy_in_ref_patch.emplace_back(0.0f);
            }
        }
    }

    hessian(1, 2) = hessian(0, 3);
    hessian(1, 4) = hessian(0, 5);
    hessian(3, 4) = hessian(2, 3);
    for (uint32_t i = 0; i < 6; ++i) {
        for (uint32_t j = i + 1; j < 6; ++j) {
            hessian(j, i) = hessian(i, j);
        }
    }
}

int32_t OpticalFlowAffineKlt::ComputeBias(const GrayImage &cur_image, const Vec2 &cur_pixel_uv, const std::vector<float> &ex_ref_patch,
                                          const std::vector<bool> &ex_ref_patch_pixel_valid, int32_t ex_ref_patch_rows, int32_t ex_ref_patch_cols,
                                          const std::vector<float> &all_dx_in_ref_patch, const std::vector<float> &all_dy_in_ref_patch, const Mat2 &affine,
                                          Vec6 &bias) {
    int32_t valid_pixel_cnt = 0;
    bias.setZero();

    for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
        for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
            // Check if the pixel in patch is valid in current image.
            const Vec2 affined_dcol_drow = affine * Vec2(dcol, drow);
            const float row_in_cur_image = affined_dcol_drow.y() + cur_pixel_uv.y();
            const float col_in_cur_image = affined_dcol_drow.x() + cur_pixel_uv.x();
            float cur_pixel_value = 0.0f;
            if (cur_image.GetPixelValue(row_in_cur_image, col_in_cur_image, &cur_pixel_value)) {
                // If valid, compute residual by precomputed ex_ref_patch in reference image.
                const int32_t row_in_ex_patch = drow + options().kPatchRowHalfSize + 1;
                const int32_t col_in_ex_patch = dcol + options().kPatchColHalfSize + 1;
                const int32_t index_in_ex_patch = row_in_ex_patch * ex_ref_patch_cols + col_in_ex_patch;

                // If the pixel is not valid in reference patch, ignore it.
                CONTINUE_IF(!ex_ref_patch_pixel_valid[index_in_ex_patch]);

                // Compute residual.
                const float ref_pixel_value = ex_ref_patch[index_in_ex_patch];
                const float dt = cur_pixel_value - ref_pixel_value;

                // Compute bias.
                const int32_t patch_cols = ex_ref_patch_cols - 2;
                const int32_t row_in_patch = row_in_ex_patch - 1;
                const int32_t col_in_patch = col_in_ex_patch - 1;
                const int32_t index_in_patch = row_in_patch * patch_cols + col_in_patch;
                const float &dx = all_dx_in_ref_patch[index_in_patch];
                const float &dy = all_dy_in_ref_patch[index_in_patch];
                bias(0) -= dt * col_in_cur_image * dx;
                bias(1) -= dt * col_in_cur_image * dy;
                bias(2) -= dt * row_in_cur_image * dx;
                bias(3) -= dt * row_in_cur_image * dy;
                bias(4) -= dt * dx;
                bias(5) -= dt * dy;

                // Statis valid pixel number.
                ++valid_pixel_cnt;
            }
        }
    }

    return valid_pixel_cnt;
}

}  // namespace feature_tracker
