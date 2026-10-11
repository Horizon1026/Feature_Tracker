#include "optical_flow_affine_klt.h"

#include "slam_operations.h"

namespace feature_tracker {

bool OpticalFlowAffineKlt::TrackMultipleLevel(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, const std::vector<Vec2> &ref_pixel_uv,
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

        // Define affine transform matrix.
        Mat2 affine = sub_options_.kPredictionAffine;

        for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
            const GrayImage &ref_image = ref_pyramid.GetImageConst(level_idx);
            const GrayImage &cur_image = cur_pyramid.GetImageConst(level_idx);

            // Track this feature in one pyramid level.
            switch (options().kMethod) {
                case OpticalFlow::Method::kInverse:
                case OpticalFlow::Method::kDirect:
                    TrackOneFeature(ref_image, cur_image, scaled_ref_pixel_uv, scaled_cur_pixel_uv, affine, status[feature_id]);
                    break;
                case OpticalFlow::Method::kFast:
                default:
                    TrackOneFeatureFast(ref_image, cur_image, scaled_ref_pixel_uv, scaled_cur_pixel_uv, affine, status[feature_id]);
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

bool OpticalFlowAffineKlt::TrackSingleLevel(const GrayImage &ref_image, const GrayImage &cur_image, const std::vector<Vec2> &ref_pixel_uv,
                                            std::vector<Vec2> &cur_pixel_uv, std::vector<uint8_t> &status) {
    // Track per feature.
    const uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ? ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
    for (uint32_t feature_id = 0; feature_id < max_feature_id; ++feature_id) {
        // Do not repeatly track features that has been tracking failed.
        CONTINUE_IF(status[feature_id] > static_cast<uint8_t>(TrackStatus::kTracked));

        // Define affine transform matrix.
        Mat2 affine = sub_options_.kPredictionAffine;

        switch (options().kMethod) {
            case OpticalFlow::Method::kInverse:
            case OpticalFlow::Method::kDirect:
                TrackOneFeature(ref_image, cur_image, ref_pixel_uv[feature_id], cur_pixel_uv[feature_id], affine, status[feature_id]);
                break;
            case OpticalFlow::Method::kFast:
            default:
                TrackOneFeatureFast(ref_image, cur_image, ref_pixel_uv[feature_id], cur_pixel_uv[feature_id], affine, status[feature_id]);
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

void OpticalFlowAffineKlt::TrackOneFeature(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv, Vec2 &cur_pixel_uv, Mat2 &affine,
                                           uint8_t &status) {
    // Clear status firstly, then try to track.
    status = static_cast<uint8_t>(TrackStatus::kLargeResidual);

    Mat6 hessian = Mat6::Zero();
    Vec6 bias = Vec6::Zero();
    for (uint32_t iter = 0; iter < options().kMaxIteration; ++iter) {
        // Construct incremental function. Statis average residual and count valid pixel.
        if (ConstructIncrementalFunction(ref_image, cur_image, ref_pixel_uv, cur_pixel_uv, affine, hessian, bias) == 0) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Validate observation.
        const float min_eig = Utility::MinEigenValue(Mat(hessian));
        if (min_eig < options().kMinEigenvalueThreshold) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Apply diagonal regularization for numerical stability of the 6x6 system.
        hessian.diagonal().array() += options().kHessianRegularizationLambda;

        // Solve hessian * dx = bias.
        const Eigen::LDLT<Mat6> ldlt_solver(hessian);
        if (ldlt_solver.info() != Eigen::Success) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }
        const Vec6 dx = ldlt_solver.solve(bias);
        if (!UpdateAffineWarp(dx, cur_pixel_uv, affine)) {
            status = static_cast<uint8_t>(TrackStatus::kNumericError);
            break;
        }

        // Check converge status.
        if (cur_pixel_uv.x() < 0 || cur_pixel_uv.x() > cur_image.cols() - 1 || cur_pixel_uv.y() < 0 || cur_pixel_uv.y() > cur_image.rows() - 1) {
            status = static_cast<uint8_t>(TrackStatus::kOutside);
            break;
        }
        if (dx.squaredNorm() < options().kMaxSquaredConvergeStep) {
            status = static_cast<uint8_t>(TrackStatus::kTracked);
            break;
        }
    }
}

bool OpticalFlowAffineKlt::UpdateAffineWarp(const Vec6 &dx, Vec2 &cur_pixel_uv, Mat2 &affine) const {
    RETURN_FALSE_IF (!dx.allFinite()) ;

    Mat2 increment = Mat2::Zero();
    increment.col(0) = dx.head<2>();
    increment.col(1) = dx.segment<2>(2);
    Mat2 updated_affine = Mat2::Zero();
    Vec2 pixel_step = Vec2::Zero();
    if (options().kMethod == OpticalFlow::Method::kDirect) {
        // Forward additive: W(d) = A*d + p, with delta W(d) = increment*d + dx.tail<2>().
        updated_affine = affine + increment;
        pixel_step = dx.tail<2>();
    } else {
        // Inverse compositional: bias = -J_ref^T*(I_cur(W(d)) - I_ref(d)), so delta = -dx.
        // Compose W with Delta^{-1}, where Delta(d) = (I - increment)*d - dx.tail<2>().
        const Mat2 delta_affine = Mat2::Identity() - increment;
        const Eigen::FullPivLU<Mat2> delta_solver(delta_affine);
        if (!delta_solver.isInvertible()) {
            return false;
        }
        updated_affine = affine * delta_solver.inverse();
        pixel_step = updated_affine * dx.tail<2>();
    }

    const Vec2 updated_pixel_uv = cur_pixel_uv + pixel_step;
    if (!updated_affine.allFinite() || !updated_pixel_uv.allFinite()) {
        return false;
    }
    affine = updated_affine;
    cur_pixel_uv = updated_pixel_uv;
    return true;
}

int32_t OpticalFlowAffineKlt::ConstructIncrementalFunction(const GrayImage &ref_image, const GrayImage &cur_image, const Vec2 &ref_pixel_uv,
                                                           const Vec2 &cur_pixel_uv, const Mat2 &affine, Mat6 &hessian, Vec6 &bias) {
    hessian.setZero();
    bias.setZero();
    std::array<float, 6> temp_value;
    int32_t num_of_valid_pixel = 0;

    if (options().kMethod == OpticalFlow::Method::kDirect) {
        // For direct optical flow, use current image to compute gradient.
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();

                Vec2 affined_dcol_drow = affine * Vec2(dcol, drow);
                const float row_j = affined_dcol_drow.y() + cur_pixel_uv.y();
                const float col_j = affined_dcol_drow.x() + cur_pixel_uv.x();

                // Compute pixel gradient.
                if (cur_image.GetPixelValue(row_j, col_j - 1.0f, &temp_value[0]) && cur_image.GetPixelValue(row_j, col_j + 1.0f, &temp_value[1]) &&
                    cur_image.GetPixelValue(row_j - 1.0f, col_j, &temp_value[2]) && cur_image.GetPixelValue(row_j + 1.0f, col_j, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) && cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float dx = 0.5f * (temp_value[1] - temp_value[0]);
                    const float dy = 0.5f * (temp_value[3] - temp_value[2]);
                    const float dt = temp_value[5] - temp_value[4];

                    // Parameters act on offsets from the reference patch center.
                    const float x = static_cast<float>(dcol);
                    const float y = static_cast<float>(drow);

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
                    hessian(3, 4) += y * dxdy;
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
        for (int32_t drow = -options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
            for (int32_t dcol = -options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
                const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();

                Vec2 affined_dcol_drow = affine * Vec2(dcol, drow);
                const float row_j = affined_dcol_drow.y() + cur_pixel_uv.y();
                const float col_j = affined_dcol_drow.x() + cur_pixel_uv.x();

                // Compute pixel gradient.
                if (ref_image.GetPixelValue(row_i, col_i - 1.0f, &temp_value[0]) && ref_image.GetPixelValue(row_i, col_i + 1.0f, &temp_value[1]) &&
                    ref_image.GetPixelValue(row_i - 1.0f, col_i, &temp_value[2]) && ref_image.GetPixelValue(row_i + 1.0f, col_i, &temp_value[3]) &&
                    ref_image.GetPixelValue(row_i, col_i, &temp_value[4]) && cur_image.GetPixelValue(row_j, col_j, &temp_value[5])) {
                    const float dx = 0.5f * (temp_value[1] - temp_value[0]);
                    const float dy = 0.5f * (temp_value[3] - temp_value[2]);
                    const float dt = temp_value[5] - temp_value[4];

                    // Parameters act on offsets from the reference patch center.
                    const float x = static_cast<float>(dcol);
                    const float y = static_cast<float>(drow);

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
                    hessian(3, 4) += y * dxdy;
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

}  // namespace feature_tracker
