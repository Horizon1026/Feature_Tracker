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
                                           const Vec2 &ref_point,
                                           Vec2 &cur_point,
                                           uint8_t &status) {
    Mat6 H = Mat6::Zero();
    Vec6 b = Vec6::Zero();
    Mat2 A = Mat2::Identity();    /* Affine trasform matrix. */

    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Construct incremental function. Statis average residual and count valid pixel.
        BREAK_IF(ConstructIncrementalFunction(ref_image, cur_image, ref_point, cur_point, A, H, b) == 0);

        // Solve H * z = b.
        const Vec6 z = H.ldlt().solve(b);
        const Vec2 v = z.head<2>() * cur_point.x() + z.segment<2>(2) * cur_point.y() + z.tail<2>();

        if (std::isnan(v(0)) || std::isnan(v(1))) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update cur_pixel_uv.
        cur_point.x() += v(0);
        cur_point.y() += v(1);

        // Update affine translation matrix.
        A.col(0) += z.head<2>();
        A.col(1) += z.segment<2>(2);

        // Check converge status.
        if (cur_point.x() < 0 || cur_point.x() > cur_image.cols() - 1 ||
            cur_point.y() < 0 || cur_point.y() > cur_image.rows() - 1) {
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
                                                           const Vec2 &ref_point,
                                                           const Vec2 &cur_point,
                                                           Mat2 &A,
                                                           Mat6 &H,
                                                           Vec6 &b) {
    H.setZero();
    b.setZero();
    std::array<float, 6> temp_value = {};
    int32_t num_of_valid_pixel = 0;

    if (options().kMethod == OpticalFlowMethod::kDirect) {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_point.y();
                const float col_i = static_cast<float>(dcol) + ref_point.x();

                Vec2 affined_dcol_drow = A * Vec2(dcol, drow);
                const float row_j = affined_dcol_drow.y() + cur_point.y();
                const float col_j = affined_dcol_drow.x() + cur_point.x();

                // Compute pixel gradient
                if (cur_image.GetPixelValue(row_j, col_j - 1.0f, &temp_value[0]) &&
                    cur_image.GetPixelValue(row_j, col_j + 1.0f, &temp_value[1]) &&
                    cur_image.GetPixelValue(row_j - 1.0f, col_j, &temp_value[2]) &&
                    cur_image.GetPixelValue(row_j + 1.0f, col_j, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) &&
                    cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float fx = temp_value[1] - temp_value[0];
                    const float fy = temp_value[3] - temp_value[2];
                    const float ft = temp_value[5] - temp_value[4];

                    const float &x = col_j;
                    const float &y = row_j;

                    const float xx = x * x;
                    const float yy = y * y;
                    const float fxfx = fx * fx;
                    const float fyfy = fy * fy;
                    const float xy = x * y;
                    const float fxfy = fx * fy;

                    H(0, 0) += xx * fxfx;
                    H(0, 1) += xx * fxfy;
                    H(0, 2) += xy * fxfx;
                    H(0, 3) += xy * fxfy;
                    H(0, 4) += x * fxfx;
                    H(0, 5) += x * fxfy;
                    H(1, 1) += xx * fyfy;
                    H(1, 2) += xy * fxfy;
                    H(1, 3) += xy * fyfy;
                    H(1, 4) += x * fxfy;
                    H(1, 5) += x * fyfy;
                    H(2, 2) += yy * fxfx;
                    H(2, 3) += yy * fxfy;
                    H(2, 4) += y * fxfx;
                    H(2, 5) += y * fxfy;
                    H(3, 3) += yy * fyfy;
                    H(3, 4) += yy * fxfy;
                    H(3, 5) += y * fyfy;
                    H(4, 4) += fxfx;
                    H(4, 5) += fxfy;
                    H(5, 5) += fyfy;

                    b(0) -= ft * x * fx;
                    b(1) -= ft * x * fy;
                    b(2) -= ft * y * fx;
                    b(3) -= ft * y * fy;
                    b(4) -= ft * fx;
                    b(5) -= ft * fy;

                    ++num_of_valid_pixel;
                }
            }
        }
    } else {
        // For inverse optical flow, use reference image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_point.y();
                const float col_i = static_cast<float>(dcol) + ref_point.x();

                Vec2 affined_dcol_drow = A * Vec2(dcol, drow);
                const float row_j = affined_dcol_drow.y() + cur_point.y();
                const float col_j = affined_dcol_drow.x() + cur_point.x();

                // Compute pixel gradient
                if (ref_image.GetPixelValue(row_i, col_i - 1.0f, &temp_value[0]) &&
                    ref_image.GetPixelValue(row_i, col_i + 1.0f, &temp_value[1]) &&
                    ref_image.GetPixelValue(row_i - 1.0f, col_i, &temp_value[2]) &&
                    ref_image.GetPixelValue(row_i + 1.0f, col_i, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) &&
                    cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float fx = temp_value[1] - temp_value[0];
                    const float fy = temp_value[3] - temp_value[2];
                    const float ft = temp_value[5] - temp_value[4];

                    const float &x = col_j;
                    const float &y = row_j;

                    const float xx = x * x;
                    const float yy = y * y;
                    const float fxfx = fx * fx;
                    const float fyfy = fy * fy;
                    const float xy = x * y;
                    const float fxfy = fx * fy;

                    H(0, 0) += xx * fxfx;
                    H(0, 1) += xx * fxfy;
                    H(0, 2) += xy * fxfx;
                    H(0, 3) += xy * fxfy;
                    H(0, 4) += x * fxfx;
                    H(0, 5) += x * fxfy;
                    H(1, 1) += xx * fyfy;
                    H(1, 2) += xy * fxfy;
                    H(1, 3) += xy * fyfy;
                    H(1, 4) += x * fxfy;
                    H(1, 5) += x * fyfy;
                    H(2, 2) += yy * fxfx;
                    H(2, 3) += yy * fxfy;
                    H(2, 4) += y * fxfx;
                    H(2, 5) += y * fxfy;
                    H(3, 3) += yy * fyfy;
                    H(3, 4) += yy * fxfy;
                    H(3, 5) += y * fyfy;
                    H(4, 4) += fxfx;
                    H(4, 5) += fxfy;
                    H(5, 5) += fyfy;

                    b(0) -= ft * x * fx;
                    b(1) -= ft * x * fy;
                    b(2) -= ft * y * fx;
                    b(3) -= ft * y * fy;
                    b(4) -= ft * fx;
                    b(5) -= ft * fy;

                    ++num_of_valid_pixel;
                }
            }
        }
    }

    for (uint32_t i = 0; i < 6; ++i) {
        for (uint32_t j = i; j < 6; ++j) {
            if (i != j) {
                H(j, i) = H(i, j);
            }
        }
    }

    return num_of_valid_pixel;
}

}
