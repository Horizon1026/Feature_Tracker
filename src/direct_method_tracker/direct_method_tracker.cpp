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
    // TODO:

    return true;
}

bool DirectMethod::TrackSingleLevel(const ImagePyramid &ref_pyramid,
                                    const ImagePyramid &cur_pyramid,
                                    const std::vector<Vec3> &p_c_in_ref,
                                    const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv,
                                    Quat &q_rc,
                                    Vec3 &p_rc) {
    // TODO:

    return true;
}

}
