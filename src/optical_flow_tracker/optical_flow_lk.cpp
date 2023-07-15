#include "optical_flow_lk.h"
#include "slam_operations.h"
#include "log_report.h"

#include "cmath"

namespace FEATURE_TRACKER {

namespace {
    static Vec3 kInfinityVec3 = Vec3(INFINITY, INFINITY, INFINITY);
}

bool OpticalFlowLk::PrepareForTracking() {
    // Initial fx_fy_ti_ and pixel_values_in_patch_ for fast inverse tracker.
    if (options().kMethod == OpticalFlowMethod::kFast) {
        const int32_t patch_rows = 2 * options().kPatchRowHalfSize + 1;
        const int32_t patch_cols = 2 * options().kPatchColHalfSize + 1;
        const uint32_t size = patch_rows + patch_cols;
        if (fx_fy_ti_.capacity() < size) {
            fx_fy_ti_.reserve(size);
        }

        pixel_values_in_patch_.resize(patch_rows + 2, patch_cols + 2);
    }

    return true;
}

bool OpticalFlowLk::TrackSingleLevel(const GrayImage &ref_image,
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

void OpticalFlowLk::PrecomputeHessian(const GrayImage &ref_image,
                                      const Vec2 &ref_point,
                                      Mat2 &H) {
    fx_fy_ti_.clear();
    Vec3 one_fx_fy_ti;

    float row_i = ref_point.y();
    float col_i = ref_point.x();
    bool no_need_check = row_i - 1.0f - options().kPatchRowHalfSize > kZero &&
                         row_i + 2.0f + options().kPatchRowHalfSize < ref_image.rows() - kZero &&
                         col_i - 1.0f - options().kPatchColHalfSize > kZero &&
                         col_i + 2.0f + options().kPatchColHalfSize < ref_image.cols() - kZero;

    int32_t row_i_buf = 0;
    int32_t col_i_buf = 0;
    pixel_values_in_patch_.setConstant(-1);

    /*
        The pixels need to compute gradient is like this:
        [ ] [x] [ ]         [ ] [x] [i] [ ]
        [x] [x] [x]   ->    [x] [o] [o] [i]
        [ ] [x] [ ]         [ ] [x] [i] [ ]
        When (drow, dcol) is moved in a small patch, the above will move one step too. Then two pixel value is computed repeatedly,
        which is shown above with 'o'.
        So, pixel_values_in_patch_ is used to avoid repeatedly computation. GetPixelValueFrameBuffer() will priorly get value from
        pixel_values_in_patch_, unless the pixel value hasn't been computed.
    */

    std::array<float, 5> temp_value = {};
    for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
        for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
            row_i = static_cast<float>(drow) + ref_point.y();
            col_i = static_cast<float>(dcol) + ref_point.x();

            if (no_need_check ||
                (row_i - 1.0f > kZero && row_i + 2.0f < ref_image.rows() - kZero &&
                 col_i - 1.0f > kZero && col_i + 2.0f < ref_image.cols() - kZero)) {

                row_i_buf = drow + options().kPatchRowHalfSize + 1;
                col_i_buf = dcol + options().kPatchColHalfSize + 1;

                // Get each pixel value in patch by linear interpolar.
                GetPixelValueFrameBuffer(ref_image, row_i_buf, col_i_buf, row_i, col_i, &temp_value[0]);
                GetPixelValueFrameBuffer(ref_image, row_i_buf + 1, col_i_buf, row_i + 1.0f, col_i, &temp_value[1]);
                GetPixelValueFrameBuffer(ref_image, row_i_buf - 1, col_i_buf, row_i - 1.0f, col_i, &temp_value[2]);
                GetPixelValueFrameBuffer(ref_image, row_i_buf, col_i_buf + 1, row_i, col_i + 1.0f, &temp_value[3]);
                GetPixelValueFrameBuffer(ref_image, row_i_buf, col_i_buf - 1, row_i, col_i - 1.0f, &temp_value[4]);

                one_fx_fy_ti(0) = temp_value[3] - temp_value[4];
                one_fx_fy_ti(1) = temp_value[1] - temp_value[2];
                one_fx_fy_ti(2) = temp_value[0];
                fx_fy_ti_.emplace_back(one_fx_fy_ti);

                const float &fx = one_fx_fy_ti.x();
                const float &fy = one_fx_fy_ti.y();

                H(0, 0) += fx * fx;
                H(1, 1) += fy * fy;
                H(0, 1) += fx * fy;

                continue;
            }

            fx_fy_ti_.emplace_back(kInfinityVec3);
        }
    }

    H(1, 0) = H(0, 1);
}

float OpticalFlowLk::ComputeResidual(const GrayImage &cur_image,
                                     const Vec2 &cur_point,
                                     Vec2 &b) {
    float residual = 0.0f;
    int32_t num_of_valid_pixel = 0;

    float row_j = cur_point.y();
    float col_j = cur_point.x();
    bool no_need_check = row_j - options().kPatchRowHalfSize > kZero &&
                         row_j + 1.0f + options().kPatchRowHalfSize < cur_image.rows() - kZero &&
                         col_j - options().kPatchColHalfSize > kZero &&
                         col_j + 1.0f + options().kPatchColHalfSize < cur_image.cols() - kZero;

    // Compute each pixel in the patch, create H * v = b.
    uint32_t idx = 0;
    for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
        for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
            row_j = static_cast<float>(drow) + cur_point.y();
            col_j = static_cast<float>(dcol) + cur_point.x();

            if ((no_need_check ||
                 (row_j > kZero && row_j + 1.0f < cur_image.rows() - kZero &&
                  col_j > kZero && col_j + 1.0f < cur_image.cols() - kZero)) &&
                 !std::isinf(fx_fy_ti_[idx].x())) {

                // Compute pixel gradient
                const float tj = cur_image.GetPixelValueNoCheck(row_j, col_j);
                const float &fx = fx_fy_ti_[idx].x();
                const float &fy = fx_fy_ti_[idx].y();
                const float ft = tj - fx_fy_ti_[idx].z();

                b(0) -= fx * ft;
                b(1) -= fy * ft;

                residual += std::fabs(ft);
                ++num_of_valid_pixel;
            }

            ++idx;
        }
    }

    if (num_of_valid_pixel) {
        residual /= static_cast<float>(num_of_valid_pixel);
    } else {
        residual = options().kMaxConvergeResidual;
    }
    return residual;
}

void OpticalFlowLk::TrackOneFeatureFast(const GrayImage &ref_image,
                                        const GrayImage &cur_image,
                                        const Vec2 &ref_point,
                                        Vec2 &cur_point,
                                        uint8_t &status) {
    // H = (A.t * A).inv * A.t.
    Mat2 H = Mat2::Zero();
    Vec2 b = Vec2::Zero();

    // Precompute H, fx, fy and ti.
    PrecomputeHessian(ref_image, ref_point, H);

    // Iterate to compute optical flow.
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        b.setZero();

        // Compute b and residual.
        const float residual = ComputeResidual(cur_image, cur_point, b);

        // Solve H * v = b, update cur_pixel_uv.
        const Vec2 v = H.ldlt().solve(b);

        if (std::isnan(v(0)) || std::isnan(v(1))) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        cur_point += v;

        if (v.squaredNorm() < options().kMaxConvergeStep) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }

        if (residual < options().kMaxConvergeResidual) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
    }

    if (cur_point.x() < 0 || cur_point.x() > cur_image.cols() - 1 ||
        cur_point.y() < 0 || cur_point.y() > cur_image.rows() - 1) {
        status = static_cast<uint8_t>(TrackStatus::kOutside);
    }
}

void OpticalFlowLk::ConstructIncrementalFunction(const GrayImage &ref_image,
                                                 const GrayImage &cur_image,
                                                 const Vec2 &ref_point,
                                                 const Vec2 &cur_point,
                                                 Mat2 &H,
                                                 Vec2 &b,
                                                 float &average_residual,
                                                 int32_t &num_of_valid_pixel) {
    std::array<float, 6> temp_value = {};

    if (options().kMethod == OpticalFlowMethod::kInverse) {
        // For inverse optical flow, use reference image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_point.y();
                const float col_i = static_cast<float>(dcol) + ref_point.x();
                const float row_j = static_cast<float>(drow) + cur_point.y();
                const float col_j = static_cast<float>(dcol) + cur_point.x();
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

                    average_residual += std::fabs(ft);
                    ++num_of_valid_pixel;
                }
            }
        }
    } else {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_point.y();
                const float col_i = static_cast<float>(dcol) + ref_point.x();
                const float row_j = static_cast<float>(drow) + cur_point.y();
                const float col_j = static_cast<float>(dcol) + cur_point.x();
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

                    average_residual += std::fabs(ft);
                    ++num_of_valid_pixel;
                }
            }
        }
    }
    H(1, 0) = H(0, 1);
    average_residual /= static_cast<float>(num_of_valid_pixel);
}

void OpticalFlowLk::TrackOneFeature(const GrayImage &ref_image,
                                    const GrayImage &cur_image,
                                    const Vec2 &ref_point,
                                    Vec2 &cur_point,
                                    uint8_t &status) {
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Compute each pixel in the patch, create H * v = b
        Mat2 H = Mat2::Zero();
        Vec2 b = Vec2::Zero();
        float average_residual = 0.0f;
        int32_t num_of_valid_pixel = 0;
        ConstructIncrementalFunction(ref_image, cur_image, ref_point, cur_point, H, b, average_residual, num_of_valid_pixel);

        // Solve H * v = b.
        Vec2 v = H.ldlt().solve(b);
        if (Eigen::isnan(v.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Update cur_pixel_uv.
        cur_point += v;

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
        if (average_residual < options().kMaxConvergeResidual && num_of_valid_pixel) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
    }
}

}
