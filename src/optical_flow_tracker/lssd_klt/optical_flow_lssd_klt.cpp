#include "optical_flow_lssd_klt.h"

#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace feature_tracker {

bool OpticalFlowLssdKlt::TrackMultipleLevel(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, const std::vector<Vec2> &ref_pixel_uv,
                                            std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) {
    const uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ? ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
    const float scale = static_cast<float>(1 << (ref_pyramid.level() - 1));

    // Track each pixel per level.
    for (uint32_t feature_id = 0; feature_id < max_feature_id; ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        CONTINUE_IF(status[feature_id] > static_cast<uint8_t>(TrackStatus::kTracked));

        // Recorder scaled ref_pixel_uv and cur_pixel_uv.
        Vec2 scaled_ref_pixel_uv = ref_pixel_uv[feature_id] / scale;
        const Vec2 scaled_cur_pixel_uv = cur_pixel_uv[feature_id] / scale;

        // Define se2 transform.
        Mat2 R_cr = sub_options_.kPredictionRcr;
        Vec2 t_cr = scaled_cur_pixel_uv - sub_options_.kPredictionRcr * scaled_ref_pixel_uv;

        for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
            const GrayImage &ref_image = ref_pyramid.GetImageConst(level_idx);
            const GrayImage &cur_image = cur_pyramid.GetImageConst(level_idx);

            // Track this feature in one pyramid level.
            switch (options().kMethod) {
                case OpticalFlow::Method::kInverse:
                case OpticalFlow::Method::kDirect:
                    TrackOneFeature(ref_image, cur_image, scaled_ref_pixel_uv, R_cr, t_cr, status[feature_id]);
                    break;
                case OpticalFlow::Method::kFast:
                default:
                    TrackOneFeatureFast(ref_image, cur_image, scaled_ref_pixel_uv, R_cr, t_cr, status[feature_id]);
                    break;
            }

            // If feature is tracked in final level, recovery its scale.
            if (!level_idx) {
                cur_pixel_uv[feature_id] = R_cr * ref_pixel_uv[feature_id] + t_cr;
                break;
            }

            // Adjust result on different pyramid level.
            scaled_ref_pixel_uv *= 2.0f;
            t_cr *= 2.0f;
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

bool OpticalFlowLssdKlt::TrackSingleLevel(const GrayImage &ref_image, const GrayImage &cur_image, const std::vector<Vec2> &ref_pixel_uv,
                                          std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) {
    // Track per feature.
    const uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ? ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
    for (uint32_t feature_id = 0; feature_id < max_feature_id; ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        CONTINUE_IF(status[feature_id] > static_cast<uint8_t>(TrackStatus::kTracked));

        // Define se2 transform.
        Mat2 R_cr = sub_options_.kPredictionRcr;
        Vec2 t_cr = cur_pixel_uv[feature_id] - sub_options_.kPredictionRcr * ref_pixel_uv[feature_id];

        switch (options().kMethod) {
            case OpticalFlow::Method::kInverse:
            case OpticalFlow::Method::kDirect:
                TrackOneFeature(ref_image, cur_image, ref_pixel_uv[feature_id], R_cr, t_cr, status[feature_id]);
                break;
            case OpticalFlow::Method::kFast:
            default:
                TrackOneFeatureFast(ref_image, cur_image, ref_pixel_uv[feature_id], R_cr, t_cr, status[feature_id]);
                break;
        }

        // Sync tracked result.
        cur_pixel_uv[feature_id] = t_cr + R_cr * ref_pixel_uv[feature_id];
        // If feature is outside, mark it.
        const auto &feature = cur_pixel_uv[feature_id];
        if (feature.x() < 0 || feature.x() > cur_image.cols() - 1 || feature.y() < 0 || feature.y() > cur_image.rows() - 1) {
            status[feature_id] = static_cast<uint8_t>(TrackStatus::kOutside);
        }
    }

    return true;
}

void OpticalFlowLssdKlt::TrackOneFeature(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Mat2 &R_cr, Vec2 &t_cr,
                                         uint8_t &status) {
    // Clear status firstly, then try to track.
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);

    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Compute each pixel in the patch, create hessian * v = bias
        Mat3 hessian = Mat3::Zero();
        Vec3 bias = Vec3::Zero();
        if (ConstructIncrementalFunction(ref_image, cur_image, ref_pixel_uv, R_cr, t_cr, hessian, bias) == 0) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Validate observation.
        const float min_eig = Utility::MinEigenValue(hessian);
        if (min_eig < options().kMinEigenvalueThreshold) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Apply diagonal regularization for numerical stability.
        hessian.diagonal().array() += options().kHessianRegularizationLambda;

        // Solve hessian * v = bias using LDLT.
        const Eigen::LDLT<Mat3> ldlt_solver(hessian);
        if (ldlt_solver.info() != Eigen::Success) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }
        const Vec3 v = ldlt_solver.solve(bias);
        if (Eigen::isnan(v.array()).any()) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        UpdateWarp(ref_pixel_uv, v, R_cr, t_cr);

        // Check converge status.
        if (v.squaredNorm() < options().kMaxSquaredConvergeStep) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
    }
}

void OpticalFlowLssdKlt::UpdateWarp(const Vec2 &ref_pixel_uv, const Vec3 &dx, Mat2 &R_cr, Vec2 &t_cr) const {
    // Parameterize W(d) = R_cr * d + center around the reference patch center.
    Vec2 center = R_cr * ref_pixel_uv + t_cr;
    const float cos_theta = std::cos(dx.x());
    const float sin_theta = std::sin(dx.x());
    Mat2 delta_R = Mat2::Zero();
    delta_R << cos_theta, -sin_theta, sin_theta, cos_theta;
    if (options().kMethod == OpticalFlow::Method::kDirect) {
        // Forward update: W_new(d) = R * Exp(theta) * d + center + translation.
        R_cr = (R_cr * delta_R).eval();
        center += dx.tail<2>();
    } else {
        // Reference-image Jacobians solve for Delta(d) = Exp(theta) * d + translation.
        // Apply its exact inverse: W_new = W o Delta^{-1}.
        R_cr = (R_cr * delta_R.transpose()).eval();
        center -= R_cr * dx.tail<2>();
    }
    t_cr = center - R_cr * ref_pixel_uv;
}

int32_t OpticalFlowLssdKlt::ConstructIncrementalFunction(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, const Mat2 &R_cr,
                                                         const Vec2 &t_cr, Mat3 &hessian, Vec3 &bias) {
    hessian.setZero();
    bias.setZero();
    std::array<float, 6> temp_value = {};
    int32_t num_of_valid_pixel = 0;

    // Compute average pixel value in reference patch and current patch.
    float ref_average_value = 0.0f;
    float cur_average_value = 0.0f;
    Vec3 mean_intensity_jacobian = Vec3::Zero();
    // Check pixel valid in reference and current patch.
    MatInt patch_pixel_valid;
    patch_pixel_valid.setConstant(options().kPatchRowHalfSize * 2 + 1, options().kPatchColHalfSize * 2 + 1, 1);

    // Precompute average value in reference and current patch.
    if (options().kMethod == OpticalFlow::Method::kInverse) {
        // For inverse optical flow, use reference image to compute gradient.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const Vec2 cur_patch_pixel_uv = R_cr * Vec2(col_i, row_i) + t_cr;
                const float row_j = cur_patch_pixel_uv.y();
                const float col_j = cur_patch_pixel_uv.x();

                if (ref_image.GetPixelValue(row_i, col_i - 1.0f, &temp_value[0]) && ref_image.GetPixelValue(row_i, col_i + 1.0f, &temp_value[1]) &&
                    ref_image.GetPixelValue(row_i - 1.0f, col_i, &temp_value[2]) && ref_image.GetPixelValue(row_i + 1.0f, col_i, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) && cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    ref_average_value += temp_value[4];
                    cur_average_value += temp_value[5];
                    const Vec2 gradient = 0.5f * Vec2(temp_value[1] - temp_value[0], temp_value[3] - temp_value[2]);
                    const Vec2 rotation_step = Vec2(-drow, dcol);
                    mean_intensity_jacobian += Vec3(gradient.dot(rotation_step), gradient.x(), gradient.y());
                    ++num_of_valid_pixel;
                } else {
                    patch_pixel_valid(drow + options().kPatchRowHalfSize, dcol + options().kPatchColHalfSize) = 0;
                }
            }
        }
    } else {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                const Vec2 cur_patch_pixel_uv = R_cr * Vec2(col_i, row_i) + t_cr;
                const float row_j = cur_patch_pixel_uv.y();
                const float col_j = cur_patch_pixel_uv.x();

                if (cur_image.GetPixelValue(row_j, col_j - 1.0f, &temp_value[0]) && cur_image.GetPixelValue(row_j, col_j + 1.0f, &temp_value[1]) &&
                    cur_image.GetPixelValue(row_j - 1.0f, col_j, &temp_value[2]) && cur_image.GetPixelValue(row_j + 1.0f, col_j, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) && cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    ref_average_value += temp_value[4];
                    cur_average_value += temp_value[5];
                    const Vec2 gradient = 0.5f * Vec2(temp_value[1] - temp_value[0], temp_value[3] - temp_value[2]);
                    const Vec2 rotation_step = (R_cr * Vec2(-drow, dcol)).eval();
                    mean_intensity_jacobian += Vec3(gradient.dot(rotation_step), gradient.x(), gradient.y());
                    ++num_of_valid_pixel;
                } else {
                    patch_pixel_valid(drow + options().kPatchRowHalfSize, dcol + options().kPatchColHalfSize) = 0;
                }
            }
        }
    }

    if (num_of_valid_pixel == 0) {
        return 0;
    }
    // Keep the common valid set fixed within this linearization, including the mean derivative.
    mean_intensity_jacobian /= static_cast<float>(num_of_valid_pixel);
    ref_average_value /= static_cast<float>(num_of_valid_pixel);
    cur_average_value /= static_cast<float>(num_of_valid_pixel);
    const float gradient_average_value = options().kMethod == OpticalFlow::Method::kInverse ? ref_average_value : cur_average_value;
    // A clamped denominator is constant, so its derivative is zero.
    if (gradient_average_value <= options().kLuminanceNormalizationEpsilon) {
        mean_intensity_jacobian.setZero();
    }
    ref_average_value = std::max(ref_average_value, options().kLuminanceNormalizationEpsilon);
    cur_average_value = std::max(cur_average_value, options().kLuminanceNormalizationEpsilon);
    const float inv_ref_average = 1.0f / ref_average_value;
    const float inv_cur_average = 1.0f / cur_average_value;
    const Vec3 normalized_mean_jacobian = mean_intensity_jacobian * (options().kMethod == OpticalFlow::Method::kInverse ? inv_ref_average : inv_cur_average);

    // Compute jacobian of se2.
    Mat2x3 jacobian_se2 = Mat2x3::Zero();
    jacobian_se2.block<2, 2>(0, 1).setIdentity();

    // Compute jacobian and residual.
    if (options().kMethod == OpticalFlow::Method::kInverse) {
        // For inverse optical flow, use reference image to compute gradient.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                if (patch_pixel_valid(drow + options().kPatchRowHalfSize, dcol + options().kPatchColHalfSize)) {
                    const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                    const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                    const Vec2 cur_patch_pixel_uv = R_cr * Vec2(col_i, row_i) + t_cr;
                    const float row_j = cur_patch_pixel_uv.y();
                    const float col_j = cur_patch_pixel_uv.x();

                    temp_value[0] = ref_image.GetPixelValueNoCheck(row_i, col_i - 1.0f);
                    temp_value[1] = ref_image.GetPixelValueNoCheck(row_i, col_i + 1.0f);
                    temp_value[2] = ref_image.GetPixelValueNoCheck(row_i - 1.0f, col_i);
                    temp_value[3] = ref_image.GetPixelValueNoCheck(row_i + 1.0f, col_i);
                    temp_value[4] = ref_image.GetPixelValueNoCheck(row_i, col_i);
                    temp_value[5] = cur_image.GetPixelValueNoCheck(row_j, col_j);

                    const Mat1x2 jacobian_pixel = 0.5f * Mat1x2(temp_value[1] - temp_value[0], temp_value[3] - temp_value[2]) * inv_ref_average;
                    // Inverse increments act in reference patch coordinates.
                    jacobian_se2.block<2, 1>(0, 0) = Vec2(-drow, dcol);
                    // d(I / mean) = dI / mean - (I / mean) * dmean / mean.
                    const Mat1x3 jacobian = jacobian_pixel * jacobian_se2 - (temp_value[4] * inv_ref_average) * normalized_mean_jacobian.transpose();
                    const Vec1 residual = Vec1(temp_value[5] * inv_cur_average - temp_value[4] * inv_ref_average);

                    hessian += jacobian.transpose() * jacobian;
                    // Inverse residual: I_cur(W(d)) - I_ref(Delta(d)); derivative is -J_ref.
                    bias += jacobian.transpose() * residual;
                }
            }
        }
    } else {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                if (patch_pixel_valid(drow + options().kPatchRowHalfSize, dcol + options().kPatchColHalfSize)) {
                    const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                    const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
                    const Vec2 cur_patch_pixel_uv = R_cr * Vec2(col_i, row_i) + t_cr;
                    const float row_j = cur_patch_pixel_uv.y();
                    const float col_j = cur_patch_pixel_uv.x();

                    temp_value[0] = cur_image.GetPixelValueNoCheck(row_j, col_j - 1.0f);
                    temp_value[1] = cur_image.GetPixelValueNoCheck(row_j, col_j + 1.0f);
                    temp_value[2] = cur_image.GetPixelValueNoCheck(row_j - 1.0f, col_j);
                    temp_value[3] = cur_image.GetPixelValueNoCheck(row_j + 1.0f, col_j);
                    temp_value[4] = ref_image.GetPixelValueNoCheck(row_i, col_i);
                    temp_value[5] = cur_image.GetPixelValueNoCheck(row_j, col_j);

                    const Mat1x2 jacobian_pixel = 0.5f * Mat1x2(temp_value[1] - temp_value[0], temp_value[3] - temp_value[2]) * inv_cur_average;
                    // Rotate local offsets while keeping the patch center fixed.
                    jacobian_se2.block<2, 1>(0, 0) = R_cr * Vec2(-drow, dcol);
                    // The current-patch mean varies with the direct warp.
                    const Mat1x3 jacobian = jacobian_pixel * jacobian_se2 - (temp_value[5] * inv_cur_average) * normalized_mean_jacobian.transpose();
                    const Vec1 residual = Vec1(temp_value[5] * inv_cur_average - temp_value[4] * inv_ref_average);

                    hessian += jacobian.transpose() * jacobian;
                    bias -= jacobian.transpose() * residual;
                }
            }
        }
    }

    return num_of_valid_pixel;
}

}  // namespace feature_tracker
