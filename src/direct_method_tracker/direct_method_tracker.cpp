#include "direct_method_tracker.h"

namespace FEATURE_TRACKER {

bool DirectMethod::TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                      const ImagePyramid &cur_pyramid,
                                      const Quat ref_q_wc,
                                      const Vec3 ref_p_wc,
                                      const std::vector<Vec3> &p_w,
                                      const std::vector<Vec2> &ref_pixel_uv,
                                      std::vector<Vec2> &cur_pixel_uv,
                                      Quat &cur_q_wc,
                                      Vec3 &cur_p_wc) {
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

    return TrackMultipleLevel(ref_pyramid, cur_pyramid, p_c_in_ref_, ref_pixel_uv, cur_pixel_uv, q_rc_, p_rc_);
}

bool DirectMethod::TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                      const ImagePyramid &cur_pyramid,
                                      const std::vector<Vec3> &p_c_in_ref,
                                      const std::vector<Vec2> &ref_pixel_uv,
                                      std::vector<Vec2> &cur_pixel_uv,
                                      Quat &q_rc,
                                      Vec3 &p_rc) {
    if (ref_pixel_uv.empty()) {
        return false;
    }
    if (cur_pyramid.level() != ref_pyramid.level()) {
        return false;
    }

    // Require for a camera model with fx, fy, cx, cy.
    if (camera_model_ == nullptr) {
        return false;
    }

    // If sizeof ref_pixel_uv is not equal to cur_pixel_uv, view it as no prediction.
    if (ref_pixel_uv.size() != cur_pixel_uv.size()) {
        cur_pixel_uv = ref_pixel_uv;
    }

    // Set predict and reference with scale.
    scaled_ref_points_.reserve(ref_pixel_uv.size());
    const float scale = static_cast<float>(1 << (ref_pyramid.level() - 1));
    for (uint32_t i = 0; i < ref_pixel_uv.size(); ++i) {
        scaled_ref_points_.emplace_back(ref_pixel_uv[i] / scale);
    }
    std::array<float, 4> K = {camera_model_->fx() / scale,
                              camera_model_->fy() / scale,
                              camera_model_->cx() / scale,
                              camera_model_->cy() / scale};

    // Track per level.
    for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
        const Image &ref_image = ref_pyramid.GetImageConst(level_idx);
        const Image &cur_image = cur_pyramid.GetImageConst(level_idx);

        TrackSingleLevel(ref_image, cur_image, K, p_c_in_ref, scaled_ref_points_, cur_pixel_uv, q_rc, p_rc);

        if (level_idx == 0) {
            break;
        }

        for (uint32_t i = 0; i < scaled_ref_points_.size(); ++i) {
            scaled_ref_points_[i] *= 2.0f;
            cur_pixel_uv[i] *= 2.0f;
        }
    }

    return true;
}

bool DirectMethod::TrackSingleLevel(const Image &ref_image,
                                    const Image &cur_image,
                                    const std::array<float, 4> &K,
                                    const std::vector<Vec3> &p_c_in_ref,
                                    const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv,
                                    Quat &q_rc,
                                    Vec3 &p_rc) {
    // Track all features together.
    switch (options().kMethod) {
        case kInverse:
        	TrackAllFeaturesInverse(ref_image, cur_image, K, p_c_in_ref, ref_pixel_uv, cur_pixel_uv, q_rc, p_rc);
            break;
        case kDirect:
         	TrackAllFeaturesDirect(ref_image, cur_image, K, p_c_in_ref, ref_pixel_uv, cur_pixel_uv, q_rc, p_rc);
            break;
        case kFast:
		default:
        	TrackAllFeaturesFast(ref_image, cur_image, K, p_c_in_ref, ref_pixel_uv, cur_pixel_uv, q_rc, p_rc);
         	break;
    }

    return true;
}


bool DirectMethod::TrackAllFeaturesInverse(const Image &ref_image,
                                 		   const Image &cur_image,
                                 		   const std::array<float, 4> &K,
                                		   const std::vector<Vec3> &p_c_in_ref,
                                	 	   const std::vector<Vec2> &ref_pixel_uv,
                                           std::vector<Vec2> &cur_pixel_uv,
                                           Quat &q_rc,
                                           Vec3 &p_rc) {

    return true;
}

bool DirectMethod::TrackAllFeaturesDirect(const Image &ref_image,
                                          const Image &cur_image,
                                          const std::array<float, 4> &K,
                                          const std::vector<Vec3> &p_c_in_ref,
                                          const std::vector<Vec2> &ref_pixel_uv,
                                          std::vector<Vec2> &cur_pixel_uv,
                                          Quat &q_rc,
                                          Vec3 &p_rc) {

    return true;
}

bool DirectMethod::TrackAllFeaturesFast(const Image &ref_image,
                                        const Image &cur_image,
                                        const std::array<float, 4> &K,
                              			const std::vector<Vec3> &p_c_in_ref,
                              			const std::vector<Vec2> &ref_pixel_uv,
                              			std::vector<Vec2> &cur_pixel_uv,
                              			Quat &q_rc,
                              			Vec3 &p_rc) {

    return true;
}

}
