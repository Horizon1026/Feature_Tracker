#include "optical_flow_lk.h"
#include "optical_flow_datatype.h"

#include <cmath>
#include <iostream>

namespace OPTICAL_FLOW {
bool OpticalFlowLk::TrackMultipleLevel(const ImagePyramid *ref_pyramid,
                                  const ImagePyramid *cur_pyramid,
                                  const std::vector<Eigen::Vector2f> &ref_points,
                                  std::vector<Eigen::Vector2f> &cur_points,
                                  std::vector<TrackStatus> &status) {
    if (cur_pyramid == nullptr || ref_pyramid == nullptr || ref_points.empty()) {
        return false;
    }
    if (cur_pyramid->level() != ref_pyramid->level()) {
        return false;
    }

    // Set predict and reference with scale.
    std::vector<Eigen::Vector2f> scaled_ref_points;
    scaled_ref_points.reserve(ref_points.size());

    int32_t scale = (2 << (ref_pyramid->level() - 1)) / 2;
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
    for (int32_t level_idx = ref_pyramid->level() - 1; level_idx > -1; --level_idx) {
        Image ref_image = ref_pyramid->GetImage(level_idx);
        Image cur_image = cur_pyramid->GetImage(level_idx);
        TrackSingleLevel(&ref_image, &cur_image, scaled_ref_points, cur_points, status);

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

bool OpticalFlowLk::TrackSingleLevel(const Image *ref_image,
                                     const Image *cur_image,
                                     const std::vector<Eigen::Vector2f> &ref_points,
                                     std::vector<Eigen::Vector2f> &cur_points,
                                     std::vector<TrackStatus> &status) {
    if (cur_image == nullptr || ref_image == nullptr || ref_points.empty()) {
        return false;
    }
    if (cur_image->image_data() == nullptr || ref_image->image_data() == nullptr) {
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
    for (uint32_t feature_id = 0; feature_id < ref_points.size(); ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        if (status[feature_id] > NOT_TRACKED) {
            continue;
        }

        switch (options_.kMethod) {
            default:
            case INVERSE_LSE:
                TrackOneFeatureInverse(ref_image, cur_image, ref_points[feature_id], cur_points[feature_id], status[feature_id]);
                break;
            case DIRECT_LSE:
                TrackOneFeatureDirect(ref_image, cur_image, ref_points[feature_id], cur_points[feature_id], status[feature_id]);
                break;
        }

        if (status[feature_id] == NOT_TRACKED) {
            status[feature_id] = LARGE_RESIDUAL;
        }
    }

    return true;
}

void OpticalFlowLk::TrackOneFeatureInverse(const Image *ref_image,
                                           const Image *cur_image,
                                           const Eigen::Vector2f &ref_point,
                                           Eigen::Vector2f &cur_point,
                                           TrackStatus &status) {
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // H = (A.t * A).inv * A.t
        Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
        Eigen::Vector2f b = Eigen::Vector2f::Zero();

        float Ixi = 0.0f;
        float Iyi = 0.0f;
        float Iti = 0.0f;
        float temp_value[6] = {0};

        float residual = 0.0f;
        int num_of_valid_pixel = 0;

        // Compute each pixel in the patch, create H * v = b
        for (int32_t drow = - options_.kPatchRowHalfSize; drow <= options_.kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options_.kPatchColHalfSize; dcol <= options_.kPatchColHalfSize; ++dcol) {
                float row_i = static_cast<float>(drow) + ref_point.y();
                float col_i = static_cast<float>(dcol) + ref_point.x();
                float row_j = static_cast<float>(drow) + cur_point.y();
                float col_j = static_cast<float>(dcol) + cur_point.x();
                // Compute pixel gradient
                if (ref_image->GetPixelValue(row_i, col_i - 1.0f, temp_value) &&
                    ref_image->GetPixelValue(row_i, col_i + 1.0f, temp_value + 1) &&
                    ref_image->GetPixelValue(row_i - 1.0f, col_i, temp_value + 2) &&
                    ref_image->GetPixelValue(row_i + 1.0f, col_i, temp_value + 3) &&
                    ref_image->GetPixelValue(row_i, col_i, temp_value + 4) &&
                    cur_image->GetPixelValue(row_j, col_j, temp_value + 5)) {
                    Ixi = temp_value[1] - temp_value[0];
                    Iyi = temp_value[3] - temp_value[2];
                    Iti = temp_value[5] - temp_value[4];

                    H(0, 0) += Ixi * Ixi;
                    H(1, 1) += Iyi * Iyi;
                    H(0, 1) += Ixi * Iyi;

                    b(0) -= Ixi * Iti;
                    b(1) -= Iyi * Iti;

                    residual += std::fabs(Iti);
                    ++num_of_valid_pixel;
                }
            }
        }
        H(1, 0) = H(0, 1);
        residual /= static_cast<float>(num_of_valid_pixel);

        // Solve H * v = b, update cur_points.
        Eigen::Vector2f v = H.inverse() * b;

        if (std::isnan(v(0)) || std::isnan(v(1))) {
            status = NUM_ERROR;
            break;
        }

        cur_point.x() += v(0);
        cur_point.y() += v(1);

        if (cur_point.x() < 0 || cur_point.x() > cur_image->cols() ||
            cur_point.y() < 0 || cur_point.y() > cur_image->rows()) {
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

void OpticalFlowLk::TrackOneFeatureDirect(const Image *ref_image,
                                          const Image *cur_image,
                                          const Eigen::Vector2f &ref_point,
                                          Eigen::Vector2f &cur_point,
                                          TrackStatus &status) {
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // H = (A.t * A).inv * A.t
        Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
        Eigen::Vector2f b = Eigen::Vector2f::Zero();

        float Ixi = 0.0f;
        float Iyi = 0.0f;
        float Iti = 0.0f;
        float temp_value[6] = {0};

        float residual = 0.0f;
        int num_of_valid_pixel = 0;

        // Compute each pixel in the patch, create H * v = b
        for (int32_t drow = - options_.kPatchRowHalfSize; drow <= options_.kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = - options_.kPatchColHalfSize; dcol <= options_.kPatchColHalfSize; ++dcol) {
                float row_i = static_cast<float>(drow) + ref_point.y();
                float col_i = static_cast<float>(dcol) + ref_point.x();
                float row_j = static_cast<float>(drow) + cur_point.y();
                float col_j = static_cast<float>(dcol) + cur_point.x();
                // Compute pixel gradient
                if (ref_image->GetPixelValue(row_i, col_i - 1.0f, temp_value) &&
                    ref_image->GetPixelValue(row_i, col_i + 1.0f, temp_value + 1) &&
                    cur_image->GetPixelValue(row_j - 1.0f, col_j, temp_value + 2) &&
                    cur_image->GetPixelValue(row_j + 1.0f, col_j, temp_value + 3) &&
                    ref_image->GetPixelValue(row_i, col_i, temp_value + 4) &&
                    cur_image->GetPixelValue(row_j, col_j, temp_value + 5)) {
                    Ixi = temp_value[1] - temp_value[0];
                    Iyi = temp_value[3] - temp_value[2];
                    Iti = temp_value[5] - temp_value[4];

                    H(0, 0) += Ixi * Ixi;
                    H(1, 1) += Iyi * Iyi;
                    H(0, 1) += Ixi * Iyi;

                    b(0) -= Ixi * Iti;
                    b(1) -= Iyi * Iti;

                    residual += std::fabs(Iti);
                    ++num_of_valid_pixel;
                }
            }
        }
        H(1, 0) = H(0, 1);
        residual /= static_cast<float>(num_of_valid_pixel);

        // Solve H * v = b, update cur_points.
        Eigen::Vector2f v = H.inverse() * b;

        if (std::isnan(v(0)) || std::isnan(v(1))) {
            status = NUM_ERROR;
            break;
        }

        cur_point.x() += v(0);
        cur_point.y() += v(1);

        if (cur_point.x() < 0 || cur_point.x() > cur_image->cols() ||
            cur_point.y() < 0 || cur_point.y() > cur_image->rows()) {
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