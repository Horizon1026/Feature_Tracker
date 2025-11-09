#include "optical_flow_basic_klt.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace FEATURE_TRACKER {

bool OpticalFlowBasicKlt::TrackMultipleLevel(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, const std::vector<Vec2> &ref_pixel_uv,
                                             std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) {
    const uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ? ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
    const float scale = static_cast<float>(1 << (ref_pyramid.level() - 1));

    // Track each pixel per level.
    for (uint32_t feature_id = 0; feature_id < max_feature_id; ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        CONTINUE_IF(status[feature_id] > static_cast<uint8_t>(TrackStatus::kTracked));

        // Recorder scaled ref_pixel_uv and cur_pixel_uv.
        Vec2 scaled_ref_pixel_uv = ref_pixel_uv[feature_id] / scale;
        Vec2 scaled_cur_pixel_uv = cur_pixel_uv[feature_id] / scale;

        for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
            const GrayImage &ref_image = ref_pyramid.GetImageConst(level_idx);
            const GrayImage &cur_image = cur_pyramid.GetImageConst(level_idx);

            // Track this feature in one pyramid level.
            switch (options().kMethod) {
                case OpticalFlowMethod::kInverse:
                case OpticalFlowMethod::kDirect:
                    TrackOneFeature(ref_image, cur_image, scaled_ref_pixel_uv, scaled_cur_pixel_uv, status[feature_id]);
                    break;
                case OpticalFlowMethod::kFast:
                default:
                    TrackOneFeatureFast(ref_image, cur_image, scaled_ref_pixel_uv, scaled_cur_pixel_uv, status[feature_id]);
                    break;
            }

            // If feature is tracked in final level, recovery its scale.
            if (!level_idx) {
                cur_pixel_uv[feature_id] = scaled_cur_pixel_uv;
                break;
            }

            // Adjust result on different pyramid level.
            scaled_ref_pixel_uv *= 2.0f;
            scaled_cur_pixel_uv *= 2.0f;
        }

        // If feature is outside, mark it.
        const auto &feature = cur_pixel_uv[feature_id];
        if (feature.x() < 0 || feature.x() > cur_pyramid.GetImageConst(0).cols() - 1 || feature.y() < 0 ||
            feature.y() > cur_pyramid.GetImageConst(0).rows() - 1) {
            status[feature_id] = static_cast<uint8_t>(TrackStatus::kOutside);
        }
    }

    return true;
}

bool OpticalFlowBasicKlt::TrackSingleLevel(const GrayImage &ref_image, const GrayImage &cur_image, const std::vector<Vec2> &ref_pixel_uv,
                                           std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) {
    // Track per feature.
    const uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ? ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
    for (uint32_t feature_id = 0; feature_id < max_feature_id; ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        CONTINUE_IF(status[feature_id] > static_cast<uint8_t>(TrackStatus::kTracked));

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

        // If feature is outside, mark it.
        const auto &feature = cur_pixel_uv[feature_id];
        if (feature.x() < 0 || feature.x() > cur_image.cols() - 1 || feature.y() < 0 || feature.y() > cur_image.rows() - 1) {
            status[feature_id] = static_cast<uint8_t>(TrackStatus::kOutside);
        }
    }

    return true;
}

void OpticalFlowBasicKlt::TrackOneFeature(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Vec2 &cur_pixel_uv,
                                          uint8_t &status) {
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Compute each pixel in the patch, create hessian * v = bias
        Mat2 hessian = Mat2::Zero();
        Vec2 bias = Vec2::Zero();
        BREAK_IF(ConstructIncrementalFunction(ref_image, cur_image, ref_pixel_uv, cur_pixel_uv, hessian, bias) == 0);

        // Solve hessian * v = bias.
        Vec2 v = hessian.ldlt().solve(bias);
        if (Eigen::isnan(v.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update cur_pixel_uv.
        cur_pixel_uv += v;

        // Check converge status.
        if (cur_pixel_uv.x() < 0 || cur_pixel_uv.x() > cur_image.cols() - 1 || cur_pixel_uv.y() < 0 || cur_pixel_uv.y() > cur_image.rows() - 1) {
            status = static_cast<uint8_t>(TrackStatus::kOutside);
            break;
        }
        if (v.squaredNorm() < options().kMaxConvergeStep) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
    }
}

int32_t OpticalFlowBasicKlt::ConstructIncrementalFunction(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv,
                                                          const Vec2 &cur_pixel_uv, Mat2 &hessian, Vec2 &bias) {
    std::array<float, 6> temp_value = {};
    int32_t num_of_valid_pixel = 0;

    if (options().kMethod == OpticalFlowMethod::kInverse) {
        // For inverse optical flow, use reference image to compute gradient.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const float row_j = static_cast<float>(drow) + cur_pixel_uv.y();
                const float col_j = static_cast<float>(dcol) + cur_pixel_uv.x();
                // Compute pixel gradient
                if (ref_image.GetPixelValue(row_i, col_i - 1.0f, &temp_value[0]) && ref_image.GetPixelValue(row_i, col_i + 1.0f, &temp_value[1]) &&
                    ref_image.GetPixelValue(row_i - 1.0f, col_i, &temp_value[2]) && ref_image.GetPixelValue(row_i + 1.0f, col_i, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) && cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float fx = temp_value[1] - temp_value[0];
                    const float fy = temp_value[3] - temp_value[2];
                    const float ft = temp_value[5] - temp_value[4];

                    hessian(0, 0) += fx * fx;
                    hessian(1, 1) += fy * fy;
                    hessian(0, 1) += fx * fy;

                    bias(0) -= fx * ft;
                    bias(1) -= fy * ft;

                    ++num_of_valid_pixel;
                }
            }
        }
    } else {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const float row_j = static_cast<float>(drow) + cur_pixel_uv.y();
                const float col_j = static_cast<float>(dcol) + cur_pixel_uv.x();
                // Compute pixel gradient
                if (cur_image.GetPixelValue(row_j, col_j - 1.0f, &temp_value[0]) && cur_image.GetPixelValue(row_j, col_j + 1.0f, &temp_value[1]) &&
                    cur_image.GetPixelValue(row_j - 1.0f, col_j, &temp_value[2]) && cur_image.GetPixelValue(row_j + 1.0f, col_j, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) && cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float fx = temp_value[1] - temp_value[0];
                    const float fy = temp_value[3] - temp_value[2];
                    const float ft = temp_value[5] - temp_value[4];

                    hessian(0, 0) += fx * fx;
                    hessian(1, 1) += fy * fy;
                    hessian(0, 1) += fx * fy;

                    bias(0) -= fx * ft;
                    bias(1) -= fy * ft;

                    ++num_of_valid_pixel;
                }
            }
        }
    }
    hessian(1, 0) = hessian(0, 1);

    return num_of_valid_pixel;
}

}  // namespace FEATURE_TRACKER
