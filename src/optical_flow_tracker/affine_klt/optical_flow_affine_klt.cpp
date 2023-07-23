#include "optical_flow_affine_klt.h"
#include "slam_operations.h"

namespace FEATURE_TRACKER {

bool OpticalFlowAffineKlt::TrackSingleLevel(const GrayImage &ref_image,
                                            const GrayImage &cur_image,
                                            const std::vector<Vec2> &ref_pixel_uv,
                                            std::vector<Vec2> &cur_pixel_uv,
                                            std::vector<uint8_t> &status) {
    // Track per feature.
    uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ? ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
    for (uint32_t feature_id = 0; feature_id < max_feature_id; ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        if (status[feature_id] > static_cast<uint8_t>(TrackStatus::kTracked)) {
            continue;
        }

        switch (options().kMethod) {
            case OpticalFlowMethod::kInverse:
            case OpticalFlowMethod::kDirect:
                TrackOneFeature(ref_image, cur_image, ref_pixel_uv[feature_id], cur_pixel_uv[feature_id], status[feature_id]);
                break;
            case OpticalFlowMethod::kFast:
            default:
                TrackOneFeatureFast(ref_image, cur_image, ref_pixel_uv[feature_id], cur_pixel_uv[feature_id], status[feature_id]);
                break;
        }

        if (status[feature_id] == static_cast<uint8_t>(TrackStatus::kNotTracked)) {
            status[feature_id] = static_cast<uint8_t>(TrackStatus::kLargeResidual);
        }
    }

    return true;
}

void OpticalFlowAffineKlt::TrackOneFeature(const GrayImage &ref_image,
                                           const GrayImage &cur_image,
                                           const Vec2 &ref_pixel_uv,
                                           Vec2 &cur_pixel_uv,
                                           uint8_t &status) {
    Mat6 hessian = Mat6::Zero();
    Vec6 bias = Vec6::Zero();
    Mat2 affine = Mat2::Identity();    /* Affine transform matrix. */

    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Construct incremental function. Statis average residual and count valid pixel.
        BREAK_IF(ConstructIncrementalFunction(ref_image, cur_image, ref_pixel_uv, cur_pixel_uv, affine, hessian, bias) == 0);

        // Solve hessian * z = bias.
        const Vec6 z = hessian.ldlt().solve(bias);
        const Vec2 v = z.head<2>() * cur_pixel_uv.x() + z.segment<2>(2) * cur_pixel_uv.y() + z.tail<2>();

        if (std::isnan(v(0)) || std::isnan(v(1))) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update cur_pixel_uv.
        cur_pixel_uv.x() += v(0);
        cur_pixel_uv.y() += v(1);

        // Update affine translation matrix.
        affine.col(0) += z.head<2>();
        affine.col(1) += z.segment<2>(2);

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

int32_t OpticalFlowAffineKlt::ConstructIncrementalFunction(const GrayImage &ref_image,
                                                           const GrayImage &cur_image,
                                                           const Vec2 &ref_pixel_uv,
                                                           const Vec2 &cur_pixel_uv,
                                                           const Mat2 &affine,
                                                           Mat6 &hessian,
                                                           Vec6 &bias) {
    hessian.setZero();
    bias.setZero();
    std::array<float, 6> temp_value;
    int32_t num_of_valid_pixel = 0;

    if (options().kMethod == OpticalFlowMethod::kDirect) {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();

                Vec2 affined_dcol_drow = affine * Vec2(dcol, drow);
                const float row_j = affined_dcol_drow.y() + cur_pixel_uv.y();
                const float col_j = affined_dcol_drow.x() + cur_pixel_uv.x();

                // Compute pixel gradient.
                if (cur_image.GetPixelValue(row_j, col_j - 1.0f, &temp_value[0]) &&
                    cur_image.GetPixelValue(row_j, col_j + 1.0f, &temp_value[1]) &&
                    cur_image.GetPixelValue(row_j - 1.0f, col_j, &temp_value[2]) &&
                    cur_image.GetPixelValue(row_j + 1.0f, col_j, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) &&
                    cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float dx = temp_value[1] - temp_value[0];
                    const float dy = temp_value[3] - temp_value[2];
                    const float dt = temp_value[5] - temp_value[4];

                    const float &x = col_j;
                    const float &y = row_j;

                    const float xx = x * x;
                    const float yy = y * y;
                    const float dxdx = dx * dx;
                    const float dydy = dy * dy;
                    const float xy = x * y;
                    const float dxdy = dx * dy;

                    hessian(0, 0) += xx * dxdx;
                    hessian(0, 1) += xx * dxdy;
                    hessian(0, 2) += xy * dxdx;
                    hessian(0, 3) += xy * dxdy;
                    hessian(0, 4) += x * dxdx;
                    hessian(0, 5) += x * dxdy;
                    hessian(1, 1) += xx * dydy;
                    hessian(1, 2) += xy * dxdy;
                    hessian(1, 3) += xy * dydy;
                    hessian(1, 4) += x * dxdy;
                    hessian(1, 5) += x * dydy;
                    hessian(2, 2) += yy * dxdx;
                    hessian(2, 3) += yy * dxdy;
                    hessian(2, 4) += y * dxdx;
                    hessian(2, 5) += y * dxdy;
                    hessian(3, 3) += yy * dydy;
                    hessian(3, 4) += yy * dxdy;
                    hessian(3, 5) += y * dydy;
                    hessian(4, 4) += dxdx;
                    hessian(4, 5) += dxdy;
                    hessian(5, 5) += dydy;

                    bias(0) -= dt * x * dx;
                    bias(1) -= dt * x * dy;
                    bias(2) -= dt * y * dx;
                    bias(3) -= dt * y * dy;
                    bias(4) -= dt * dx;
                    bias(5) -= dt * dy;

                    ++num_of_valid_pixel;
                }
            }
        }
    } else {
        // For inverse optical flow, use reference image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();

                Vec2 affined_dcol_drow = affine * Vec2(dcol, drow);
                const float row_j = affined_dcol_drow.y() + cur_pixel_uv.y();
                const float col_j = affined_dcol_drow.x() + cur_pixel_uv.x();

                // Compute pixel gradient.
                if (ref_image.GetPixelValue(row_i, col_i - 1.0f, &temp_value[0]) &&
                    ref_image.GetPixelValue(row_i, col_i + 1.0f, &temp_value[1]) &&
                    ref_image.GetPixelValue(row_i - 1.0f, col_i, &temp_value[2]) &&
                    ref_image.GetPixelValue(row_i + 1.0f, col_i, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) &&
                    cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float dx = temp_value[1] - temp_value[0];
                    const float dy = temp_value[3] - temp_value[2];
                    const float dt = temp_value[5] - temp_value[4];

                    const float &x = col_j;
                    const float &y = row_j;

                    const float xx = x * x;
                    const float yy = y * y;
                    const float dxdx = dx * dx;
                    const float dydy = dy * dy;
                    const float xy = x * y;
                    const float dxdy = dx * dy;

                    hessian(0, 0) += xx * dxdx;
                    hessian(0, 1) += xx * dxdy;
                    hessian(0, 2) += xy * dxdx;
                    hessian(0, 3) += xy * dxdy;
                    hessian(0, 4) += x * dxdx;
                    hessian(0, 5) += x * dxdy;
                    hessian(1, 1) += xx * dydy;
                    hessian(1, 2) += xy * dxdy;
                    hessian(1, 3) += xy * dydy;
                    hessian(1, 4) += x * dxdy;
                    hessian(1, 5) += x * dydy;
                    hessian(2, 2) += yy * dxdx;
                    hessian(2, 3) += yy * dxdy;
                    hessian(2, 4) += y * dxdx;
                    hessian(2, 5) += y * dxdy;
                    hessian(3, 3) += yy * dydy;
                    hessian(3, 4) += yy * dxdy;
                    hessian(3, 5) += y * dydy;
                    hessian(4, 4) += dxdx;
                    hessian(4, 5) += dxdy;
                    hessian(5, 5) += dydy;

                    bias(0) -= dt * x * dx;
                    bias(1) -= dt * x * dy;
                    bias(2) -= dt * y * dx;
                    bias(3) -= dt * y * dy;
                    bias(4) -= dt * dx;
                    bias(5) -= dt * dy;

                    ++num_of_valid_pixel;
                }
            }
        }
    }

    for (uint32_t i = 0; i < 6; ++i) {
        for (uint32_t j = i; j < 6; ++j) {
            if (i != j) {
                hessian(j, i) = hessian(i, j);
            }
        }
    }

    return num_of_valid_pixel;
}

}
