#include "optical_flow_klt.h"
#include "optical_flow_datatype.h"

#include <cmath>

#include "slam_operations.h"

namespace OPTICAL_FLOW {
namespace {
inline static Vec3 kInfinityVec3 = Vec3(INFINITY, INFINITY, INFINITY);
}

bool OpticalFlowKlt::TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                        const ImagePyramid &cur_pyramid,
                                        const std::vector<Eigen::Vector2f> &ref_points,
                                        std::vector<Eigen::Vector2f> &cur_points,
                                        std::vector<TrackStatus> &status) {
    if (ref_points.empty()) {
        return false;
    }
    if (cur_pyramid.level() != ref_pyramid.level()) {
        return false;
    }

    // Initial fx_fy_ti_ for fast inverse tracker.
    if (options_.kMethod == KLT_FAST) {
        const int32_t patch_rows = 2 * options_.kPatchRowHalfSize + 1;
        const int32_t patch_cols = 2 * options_.kPatchColHalfSize + 1;
        const uint32_t size = patch_rows + patch_cols;
        if (fx_fy_ti_.capacity() < size) {
            fx_fy_ti_.reserve(size);
        }
    }

    // Set predict and reference with scale.
    std::vector<Eigen::Vector2f> scaled_ref_points;
    scaled_ref_points.reserve(ref_points.size());

    int32_t scale = (2 << (ref_pyramid.level() - 1)) / 2;
    for (uint32_t i = 0; i < ref_points.size(); ++i) {
        scaled_ref_points.emplace_back(ref_points[i] / static_cast<float>(scale));
    }

    // If sizeof ref_points is not equal to cur_points, view it as no prediction.
    if (scaled_ref_points.size() != cur_points.size()) {
        cur_points = scaled_ref_points;
    } else {
        for (uint32_t i = 0; i < cur_points.size(); ++i) {
            cur_points[i] /= static_cast<float>(scale);
        }
    }

    // If sizeof ref_points is not equal to status, view it as all features haven't been tracked.
    if (scaled_ref_points.size() != status.size()) {
        status.resize(scaled_ref_points.size(), NOT_TRACKED);
    }

    // Track per level.
    for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
        Image ref_image = ref_pyramid.GetImage(level_idx);
        Image cur_image = cur_pyramid.GetImage(level_idx);
        TrackSingleLevel(ref_image, cur_image, scaled_ref_points, cur_points, status);

        if (level_idx == 0) {
            break;
        }

        for (uint32_t i = 0; i < scaled_ref_points.size(); ++i) {
            scaled_ref_points[i] *= 2.0f;
            cur_points[i] *= 2.0f;
        }
    }

    return true;
}

bool OpticalFlowKlt::TrackSingleLevel(const Image &ref_image,
                                      const Image &cur_image,
                                      const std::vector<Eigen::Vector2f> &ref_points,
                                      std::vector<Eigen::Vector2f> &cur_points,
                                      std::vector<TrackStatus> &status) {
    if (ref_points.empty()) {
        return false;
    }
    if (cur_image.data() == nullptr || ref_image.data() == nullptr) {
        return false;
    }

    // If sizeof ref_points is not equal to cur_points, view it as no prediction.
    if (ref_points.size() != cur_points.size()) {
        cur_points = ref_points;
    }

    // If sizeof ref_points is not equal to status, view it as all features haven't been tracked.
    if (ref_points.size() != status.size()) {
        status.resize(ref_points.size(), NOT_TRACKED);
    }

    // Track per feature.
    uint32_t max_feature_id = ref_points.size() < options_.kMaxTrackPointsNumber ? ref_points.size() : options_.kMaxTrackPointsNumber;
    for (uint32_t feature_id = 0; feature_id < max_feature_id; ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        if (status[feature_id] > NOT_TRACKED) {
            continue;
        }

        switch (options_.kMethod) {
            default:
            case KLT_INVERSE:
                TrackOneFeatureInverse(ref_image, cur_image, ref_points[feature_id], cur_points[feature_id], status[feature_id]);
                break;
            case KLT_DIRECT:
                TrackOneFeatureDirect(ref_image, cur_image, ref_points[feature_id], cur_points[feature_id], status[feature_id]);
                break;
        }

        if (status[feature_id] == NOT_TRACKED) {
            status[feature_id] = LARGE_RESIDUAL;
        }
    }

    return true;
}

void OpticalFlowKlt::TrackOneFeatureInverse(const Image &ref_image,
                                            const Image &cur_image,
                                            const Eigen::Vector2f &ref_point,
                                            Eigen::Vector2f &cur_point,
                                            TrackStatus &status) {
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix2f A = Eigen::Matrix2f::Identity();    /* Affine trasform matrix. */

    // Precompute H, fx, fy and ti.
    fx_fy_ti_.clear();
    float temp_value[6] = { 0 };

    for (int32_t drow = - options_.kPatchRowHalfSize; drow <= options_.kPatchRowHalfSize; ++drow) {
        for (int32_t dcol = - options_.kPatchColHalfSize; dcol <= options_.kPatchColHalfSize; ++dcol) {
            float row_i = static_cast<float>(drow) + ref_point.y();
            float col_i = static_cast<float>(dcol) + ref_point.x();
            float row_j = static_cast<float>(drow) + cur_point.y();
            float col_j = static_cast<float>(dcol) + cur_point.x();

            // Compute pixel gradient
            if (ref_image.GetPixelValue(row_i, col_i - 1.0f, temp_value) &&
                ref_image.GetPixelValue(row_i, col_i + 1.0f, temp_value + 1) &&
                ref_image.GetPixelValue(row_i - 1.0f, col_i, temp_value + 2) &&
                ref_image.GetPixelValue(row_i + 1.0f, col_i, temp_value + 3) &&
                ref_image.GetPixelValue(row_i, col_i, temp_value + 4)) {
                fx_fy_ti_.emplace_back(Eigen::Vector3f(temp_value[1] - temp_value[0],
                                                       temp_value[3] - temp_value[2],
                                                       temp_value[4]));

                const float &fx = fx_fy_ti_.back().x();
                const float &fy = fx_fy_ti_.back().y();
                const float &x = col_j;
                const float &y = row_j;

                const float xx = x * x;
                const float yy = y * y;
                const float xy = x * y;
                const float fxfx = fx * fx;
                const float fyfy = fy * fy;
                const float fxfy = fx * fy;

                H(0, 0) += xx * fxfx;
                H(0, 1) += xx * fxfy;
                H(0, 2) += xy * fxfx;
                H(0, 3) += xy * fxfy;
                H(0, 4) += x * fxfx;
                H(0, 5) += x * fxfy;
                H(1, 1) += xx * fyfy;
                H(1, 3) += xy * fyfy;
                H(1, 5) += x * fyfy;
                H(2, 2) += yy * fxfx;
                H(2, 3) += yy * fxfy;
                H(2, 4) += y * fxfx;
                H(2, 5) += y * fxfy;
                H(3, 3) += yy * fyfy;
                H(3, 5) += y * fyfy;
                H(4, 4) += fxfx;
                H(4, 5) += fxfy;
                H(5, 5) += fyfy;
            } else {
                fx_fy_ti_.emplace_back(kInfinityVec3);
            }
        }
    }

    H(1, 2) = H(0, 3);
    H(1, 4) = H(0, 5);
    H(3, 4) = H(2, 3);
    for (uint32_t i = 0; i < 6; ++i) {
        for (uint32_t j = i; j < 6; ++j) {
            if (i != j) {
                H(j, i) = H(i, j);
            }
        }
    }

    // Iterate to compute optical flow.
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        b.setZero();

        float ft = 0.0f;
        float residual = 0.0f;
        int num_of_valid_pixel = 0;

        // Compute each pixel in the patch, create H * v = b
        uint32_t idx = 0;
        for (int32_t drow = - options_.kPatchRowHalfSize; drow <= options_.kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options_.kPatchColHalfSize; dcol <= options_.kPatchColHalfSize; ++dcol) {
                Eigen::Vector2f affined_dcol_drow = A * Eigen::Vector2f(dcol, drow);
                float row_j = affined_dcol_drow.y() + cur_point.y();
                float col_j = affined_dcol_drow.x() + cur_point.x();

                // Compute pixel gradient
                if (cur_image.GetPixelValue(row_j, col_j, temp_value + 5) &&
                    !std::isinf(fx_fy_ti_[idx].x())) {
                    const float fx = fx_fy_ti_[idx].x();
                    const float fy = fx_fy_ti_[idx].y();
                    ft = temp_value[5] - fx_fy_ti_[idx].z();

                    float &x = col_j;
                    float &y = row_j;

                    b(0) -= ft * x * fx;
                    b(1) -= ft * x * fy;
                    b(2) -= ft * y * fx;
                    b(3) -= ft * y * fy;
                    b(4) -= ft * fx;
                    b(5) -= ft * fy;

                    residual += std::fabs(ft);
                    ++num_of_valid_pixel;
                }

                ++idx;
            }
        }

        residual /= static_cast<float>(num_of_valid_pixel);

        // Solve H * z = b, update cur_points.
        Eigen::Matrix<float, 6, 1> z = H.ldlt().solve(b);
        Eigen::Vector2f v = z.head<2>() * cur_point.x() + z.segment<2>(2) * cur_point.y() + z.tail<2>();

        if (std::isnan(v(0)) || std::isnan(v(1))) {
            status = NUM_ERROR;
            break;
        }

        cur_point.x() += v(0);
        cur_point.y() += v(1);

        // Update affine translation matrix.
        A.col(0) += z.head<2>();
        A.col(1) += z.segment<2>(2);

        if (cur_point.x() < 0 || cur_point.x() > cur_image.cols() ||
            cur_point.y() < 0 || cur_point.y() > cur_image.rows()) {
            status = OUTSIDE;
            break;
        }

        if (v.squaredNorm() < options_.kMaxConvergeStep) {
            status = TRACKED;
            break;
        }

        if (residual < options_.kMaxConvergeResidual && num_of_valid_pixel) {
            status = TRACKED;
            break;
        }
    }
}

void OpticalFlowKlt::TrackOneFeatureDirect(const Image &ref_image,
                                           const Image &cur_image,
                                           const Eigen::Vector2f &ref_point,
                                           Eigen::Vector2f &cur_point,
                                           TrackStatus &status) {
    Eigen::Matrix<float, 6, 6> H;
    Eigen::Matrix<float, 6, 1> b;

    Eigen::Matrix2f A = Eigen::Matrix2f::Identity();    /* Affine trasform matrix. */

    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        H.setZero();
        b.setZero();

        float fx = 0.0f;
        float fy = 0.0f;
        float ft = 0.0f;
        float temp_value[6] = { 0 };

        float residual = 0.0f;
        int num_of_valid_pixel = 0;

        // Compute each pixel in the patch, create H * v = b
        for (int32_t drow = - options_.kPatchRowHalfSize; drow <= options_.kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options_.kPatchColHalfSize; dcol <= options_.kPatchColHalfSize; ++dcol) {
                float row_i = static_cast<float>(drow) + ref_point.y();
                float col_i = static_cast<float>(dcol) + ref_point.x();

                Eigen::Vector2f affined_dcol_drow = A * Eigen::Vector2f(dcol, drow);
                float row_j = affined_dcol_drow.y() + cur_point.y();
                float col_j = affined_dcol_drow.x() + cur_point.x();

                // Compute pixel gradient
                if (cur_image.GetPixelValue(row_j, col_j - 1.0f, temp_value) &&
                    cur_image.GetPixelValue(row_j, col_j + 1.0f, temp_value + 1) &&
                    cur_image.GetPixelValue(row_j - 1.0f, col_j, temp_value + 2) &&
                    cur_image.GetPixelValue(row_j + 1.0f, col_j, temp_value + 3) &&
                    ref_image.GetPixelValue(row_i, col_i, temp_value + 4) &&
                    cur_image.GetPixelValue(row_j, col_j, temp_value + 5)) {
                    fx = temp_value[1] - temp_value[0];
                    fy = temp_value[3] - temp_value[2];
                    ft = temp_value[5] - temp_value[4];

                    float &x = col_j;
                    float &y = row_j;

                    float xx = x * x;
                    float yy = y * y;
                    float fxfx = fx * fx;
                    float fyfy = fy * fy;
                    float xy = x * y;
                    float fxfy = fx * fy;

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

                    residual += std::fabs(ft);
                    ++num_of_valid_pixel;
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

        residual /= static_cast<float>(num_of_valid_pixel);

        // Solve H * z = b, update cur_points.
        Eigen::Matrix<float, 6, 1> z = H.ldlt().solve(b);
        Eigen::Vector2f v = z.head<2>() * cur_point.x() + z.segment<2>(2) * cur_point.y() + z.tail<2>();

        if (std::isnan(v(0)) || std::isnan(v(1))) {
            status = NUM_ERROR;
            break;
        }

        cur_point.x() += v(0);
        cur_point.y() += v(1);

        // Update affine translation matrix.
        A.col(0) += z.head<2>();
        A.col(1) += z.segment<2>(2);

        if (cur_point.x() < 0 || cur_point.x() > cur_image.cols() ||
            cur_point.y() < 0 || cur_point.y() > cur_image.rows()) {
            status = OUTSIDE;
            break;
        }

        if (v.squaredNorm() < options_.kMaxConvergeStep) {
            status = TRACKED;
            break;
        }

        if (residual < options_.kMaxConvergeResidual && num_of_valid_pixel) {
            status = TRACKED;
            break;
        }
    }
}

}
