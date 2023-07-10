#include "direct_method_tracker.h"
#include "camera_basic.h"
#include "slam_operations.h"
#include "log_report.h"

namespace FEATURE_TRACKER {

bool DirectMethod::TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                      const ImagePyramid &cur_pyramid,
                                      const std::array<float, 4> &K,
                                      const Quat ref_q_wc,
                                      const Vec3 ref_p_wc,
                                      const std::vector<Vec3> &p_w,
                                      const std::vector<Vec2> &ref_pixel_uv,
                                      std::vector<Vec2> &cur_pixel_uv,
                                      Quat &cur_q_wc,
                                      Vec3 &cur_p_wc,
                                      std::vector<uint8_t> &status) {
    // Lift all points in world frame to reference camera frame.
    if (p_c_in_ref_.capacity() < p_w.size()) {
        p_c_in_ref_.reserve(p_w.size());
    }
    p_c_in_ref_.clear();

    const Quat ref_q_cw = ref_q_wc.inverse();
    for (const auto &pos_w : p_w) {
        p_c_in_ref_.emplace_back(ref_q_cw * (pos_w - ref_p_wc));
    }

    // T_rc = T_wr.inverse() * T_wc
    // [ R_rc  t_rc ] = [ R_wr.t  -R_wr.t * t_wr ] * [ R_wc  t_wc ] = [ R_wr.t * R_wc  R_wr.t * (t_wc - t_wr) ]
    // [  0      1  ]   [    0           1       ]   [  0      1  ]   [       0                  1            ]
    q_rc_ = ref_q_cw * cur_q_wc;
    p_rc_ = ref_q_cw * (cur_p_wc - ref_p_wc);

    RETURN_FALSE_IF_FALSE(TrackMultipleLevel(ref_pyramid, cur_pyramid, K, p_c_in_ref_, ref_pixel_uv, cur_pixel_uv, q_rc_, p_rc_, status));

    cur_q_wc = ref_q_wc * q_rc_;
    cur_p_wc = ref_q_wc * p_rc_ + ref_p_wc;
    return true;
}

bool DirectMethod::TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                      const ImagePyramid &cur_pyramid,
                                      const std::array<float, 4> &K,
                                      const std::vector<Vec3> &p_c_in_ref,
                                      const std::vector<Vec2> &ref_pixel_uv,
                                      std::vector<Vec2> &cur_pixel_uv,
                                      Quat &q_rc,
                                      Vec3 &p_rc,
                                      std::vector<uint8_t> &status) {
    if (ref_pixel_uv.empty()) {
        return false;
    }
    if (cur_pyramid.level() != ref_pyramid.level()) {
        return false;
    }

    // If sizeof ref_pixel_uv is not equal to cur_pixel_uv, view it as no prediction.
    if (ref_pixel_uv.size() != cur_pixel_uv.size()) {
        cur_pixel_uv = ref_pixel_uv;
    }

    // Set predict and reference with scale.
    scaled_ref_points_.clear();
    scaled_ref_points_.reserve(ref_pixel_uv.size());
    const float scale = static_cast<float>(1 << (ref_pyramid.level() - 1));
    for (uint32_t i = 0; i < ref_pixel_uv.size(); ++i) {
        scaled_ref_points_.emplace_back(ref_pixel_uv[i] / scale);
    }
    std::array<float, 4> scaled_K = { K[0] / scale, K[1] / scale, K[2] / scale, K[3] / scale };

    // Track per level.
    for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
        const GrayImage &ref_image = ref_pyramid.GetImageConst(level_idx);
        const GrayImage &cur_image = cur_pyramid.GetImageConst(level_idx);

        TrackSingleLevel(ref_image, cur_image, scaled_K, p_c_in_ref, scaled_ref_points_, cur_pixel_uv, q_rc, p_rc);

        if (level_idx == 0) {
            break;
        }

        // Recover scale.
        for (uint32_t i = 0; i < scaled_ref_points_.size(); ++i) {
            scaled_ref_points_[i] *= 2.0f;
        }
        for (uint32_t i = 0; i < 4; ++i) {
            scaled_K[i] *= 2.0f;
        }
    }

    // Check if outside.
    if (status.size() != ref_pixel_uv.size()) {
        status.resize(ref_pixel_uv.size(), static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    }
    const GrayImage &bottom_image = ref_pyramid.GetImageConst(0);
    for (uint32_t i = 0; i < cur_pixel_uv.size(); ++i) {
        if (cur_pixel_uv[i].x() < 0 || cur_pixel_uv[i].x() > bottom_image.cols() - 1 ||
            cur_pixel_uv[i].y() < 0 || cur_pixel_uv[i].y() > bottom_image.rows() - 1) {
            status[i] = static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kOutside);
        }
    }

    return true;
}

bool DirectMethod::TrackSingleLevel(const GrayImage &ref_image,
                                    const GrayImage &cur_image,
                                    const std::array<float, 4> &K,
                                    const std::vector<Vec3> &p_c_in_ref,
                                    const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv,
                                    Quat &q_rc,
                                    Vec3 &p_rc) {
    // Track all features together.
    switch (options().kMethod) {
        case kInverse:
        	RETURN_FALSE_IF_FALSE(TrackAllFeaturesInverse(ref_image, cur_image, K, p_c_in_ref, ref_pixel_uv, cur_pixel_uv, q_rc, p_rc));
            break;
        case kDirect:
         	RETURN_FALSE_IF_FALSE(TrackAllFeaturesDirect(ref_image, cur_image, K, p_c_in_ref, ref_pixel_uv, cur_pixel_uv, q_rc, p_rc));
            break;
        case kFast:
		default:
        	RETURN_FALSE_IF_FALSE(TrackAllFeaturesFast(ref_image, cur_image, K, p_c_in_ref, ref_pixel_uv, cur_pixel_uv, q_rc, p_rc));
         	break;
    }

    return true;
}


bool DirectMethod::TrackAllFeaturesInverse(const GrayImage &ref_image,
                                           const GrayImage &cur_image,
                                           const std::array<float, 4> &K,
                                           const std::vector<Vec3> &p_c_in_ref,
                                           const std::vector<Vec2> &ref_pixel_uv,
                                           std::vector<Vec2> &cur_pixel_uv,
                                           Quat &q_rc,
                                           Vec3 &p_rc) {

    return true;
}

bool DirectMethod::TrackAllFeaturesDirect(const GrayImage &ref_image,
                                          const GrayImage &cur_image,
                                          const std::array<float, 4> &K,
                                          const std::vector<Vec3> &p_c_in_ref,
                                          const std::vector<Vec2> &ref_pixel_uv,
                                          std::vector<Vec2> &cur_pixel_uv,
                                          Quat &q_rc,
                                          Vec3 &p_rc) {
    // Construct camera model with K.
    SENSOR_MODEL::CameraBasic camera(K[0], K[1], K[2], K[3]);

    // Iterate to estimate q_rc and p_rc.
    for (uint32_t iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Prepare for constructing incremental function.
        Mat6 H = Mat6::Zero();
        Vec6 b = Vec6::Zero();

        // Use all features to construct incremental function.
        uint32_t max_feature_id = ref_pixel_uv.size() < options().kMaxTrackPointsNumber ? ref_pixel_uv.size() : options().kMaxTrackPointsNumber;
        for (uint32_t i = 0; i < max_feature_id; ++i) {
            if (p_c_in_ref[i].z() < kZero) {
                continue;
            }
            const float p_r_x = p_c_in_ref[i].x();
            const float p_r_y = p_c_in_ref[i].y();
            const float p_r_z = p_c_in_ref[i].z();
            const float p_r_z_inv = 1.0f / p_r_z;
            const float p_r_z2_inv = p_r_z_inv * p_r_z_inv;
            const float fx = K[0];
            const float fy = K[1];

            // Project points to current frame.
            const Vec3 p_c_in_cur = q_rc.inverse() * (p_c_in_ref[i] - p_rc);
            if (p_c_in_cur.z() < kZero) {
                continue;
            }
            const Vec2 cur_norm_xy = (p_c_in_cur / p_c_in_cur.z()).head<2>();
            camera.LiftToImagePlane(cur_norm_xy, cur_pixel_uv[i]);

            // Compute gradient from pixel to xi.
            Mat2x6 jacobian_pixel_xi;
            jacobian_pixel_xi << fx * p_r_z_inv,
                                 0,
                                 -fx * p_r_x * p_r_z2_inv,
                                 -fx * p_r_x * p_r_y * p_r_z2_inv,
                                 fx + fx * p_r_x * p_r_x * p_r_z2_inv,
                                 -fx * p_r_y * p_r_z_inv,
                                 0,
                                 fy * p_r_z_inv,
                                 -fy * p_r_y * p_r_z2_inv,
                                 -fy - fy * p_r_y * p_r_y * p_r_z2_inv,
                                 fy * p_r_x * p_r_y * p_r_z2_inv,
                                 fy * p_r_x * p_r_z_inv;

            // Compute image gradient with all pixel in the patch, create H * v = b
            float temp_value[6] = {};
            for (int32_t drow = - options().kPatchRowHalfSize; drow <= options().kPatchRowHalfSize; ++drow) {
                for (int32_t dcol = - options().kPatchColHalfSize; dcol <= options().kPatchColHalfSize; ++dcol) {
                    const float row_i = static_cast<float>(drow) + ref_pixel_uv[i].y();
                    const float col_i = static_cast<float>(dcol) + ref_pixel_uv[i].x();
                    const float row_j = static_cast<float>(drow) + cur_pixel_uv[i].y();
                    const float col_j = static_cast<float>(dcol) + cur_pixel_uv[i].x();
                    // Compute pixel gradient
                    if (cur_image.GetPixelValue(row_j, col_j - 1.0f, temp_value) &&
                        cur_image.GetPixelValue(row_j, col_j + 1.0f, temp_value + 1) &&
                        cur_image.GetPixelValue(row_j - 1.0f, col_j, temp_value + 2) &&
                        cur_image.GetPixelValue(row_j + 1.0f, col_j, temp_value + 3) &&
                        ref_image.GetPixelValue(row_i, col_i, temp_value + 4) &&
                        cur_image.GetPixelValue(row_j, col_j, temp_value + 5)) {

                        const Vec2 jacobian_image_pixel = Vec2(temp_value[1] - temp_value[0], temp_value[3] - temp_value[2]) * 0.5f;
                        const float residual = temp_value[5] - temp_value[4];

                        // Construct full jacobian. Then use it to construct incremental function.
                        const Vec6 jacobian = (jacobian_image_pixel.transpose() * jacobian_pixel_xi).transpose();
                        H += jacobian * jacobian.transpose();
                        b += residual * jacobian;
                    }
                }
            }
        }

        // Solve incremental function.
        Vec6 dx = H.ldlt().solve(b);
        if (Eigen::isnan(dx.array()).any()) {
            break;
        }

        // Update current frame pose.
        p_rc += dx.head<3>();
        q_rc = Quat(1.0f, dx(3) * 0.5f, dx(4) * 0.5f, dx(5) * 0.5f).normalized() * q_rc;
        q_rc.normalize();

        // Check if converged.
        if (dx.squaredNorm() < options().kMaxConvergeStep) {
            break;
        }
    }

    return true;
}

bool DirectMethod::TrackAllFeaturesFast(const GrayImage &ref_image,
                                        const GrayImage &cur_image,
                                        const std::array<float, 4> &K,
                              			const std::vector<Vec3> &p_c_in_ref,
                              			const std::vector<Vec2> &ref_pixel_uv,
                              			std::vector<Vec2> &cur_pixel_uv,
                              			Quat &q_rc,
                              			Vec3 &p_rc) {

    return true;
}

}
