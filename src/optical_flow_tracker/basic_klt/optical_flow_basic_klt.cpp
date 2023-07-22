#include "optical_flow_basic_klt.h"
#include "slam_operations.h"
#include "log_report.h"

namespace FEATURE_TRACKER {

bool OpticalFlowBasicKlt::TrackSingleLevel(const GrayImage &ref_image,
                                           const GrayImage &cur_image,
                                           const std::vector<Vec2> &ref_pixel_uv,
                                           std::vector<Vec2> &cur_pixel_uv,
                                           std::vector<uint8_t> &status) {
    // Track per feature.
    const uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ?
                                    ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
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

int32_t OpticalFlowBasicKlt::ConstructIncrementalFunction(const GrayImage &ref_image,
                                                          const GrayImage &cur_image,
                                                          const Vec2 &ref_pixel_uv,
                                                          const Vec2 &cur_pixel_uv,
                                                          Mat2 &H,
                                                          Vec2 &b) {
    std::array<float, 6> temp_value = {};
    int32_t num_of_valid_pixel = 0;

    if (options().kMethod == OpticalFlowMethod::kInverse) {
        // For inverse optical flow, use reference image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const float row_j = static_cast<float>(drow) + cur_pixel_uv.y();
                const float col_j = static_cast<float>(dcol) + cur_pixel_uv.x();
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

                    H(0, 0) += fx * fx;
                    H(1, 1) += fy * fy;
                    H(0, 1) += fx * fy;

                    b(0) -= fx * ft;
                    b(1) -= fy * ft;

                    ++num_of_valid_pixel;
                }
            }
        }
    } else {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const float row_j = static_cast<float>(drow) + cur_pixel_uv.y();
                const float col_j = static_cast<float>(dcol) + cur_pixel_uv.x();
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

                    H(0, 0) += fx * fx;
                    H(1, 1) += fy * fy;
                    H(0, 1) += fx * fy;

                    b(0) -= fx * ft;
                    b(1) -= fy * ft;

                    ++num_of_valid_pixel;
                }
            }
        }
    }
    H(1, 0) = H(0, 1);

    return num_of_valid_pixel;
}

void OpticalFlowBasicKlt::TrackOneFeature(const GrayImage &ref_image,
                                          const GrayImage &cur_image,
                                          const Vec2 &ref_pixel_uv,
                                          Vec2 &cur_pixel_uv,
                                          uint8_t &status) {
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Compute each pixel in the patch, create H * v = b
        Mat2 H = Mat2::Zero();
        Vec2 b = Vec2::Zero();
        BREAK_IF(ConstructIncrementalFunction(ref_image, cur_image, ref_pixel_uv, cur_pixel_uv, H, b) == 0);

        // Solve H * v = b.
        Vec2 v = H.ldlt().solve(b);
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

}
