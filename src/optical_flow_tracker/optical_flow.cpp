#include "optical_flow.h"

namespace FEATURE_TRACKER {

bool OpticalFlow::TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                     const ImagePyramid &cur_pyramid,
                                     const std::vector<Vec2> &ref_pixel_uv,
                                     std::vector<Vec2> &cur_pixel_uv,
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

    // If sizeof ref_pixel_uv is not equal to status, view it as all features haven't been tracked.
    if (ref_pixel_uv.size() != status.size()) {
        status.resize(ref_pixel_uv.size(), static_cast<uint8_t>(TrackStatus::kNotTracked));
    }

    // Prepare for tracking.
    PrepareForTracking();

    // Set predict and reference with scale.
    scaled_ref_points_.resize(ref_pixel_uv.size());
    const float scale = static_cast<float>(1 << (ref_pyramid.level() - 1));
    for (uint32_t i = 0; i < ref_pixel_uv.size(); ++i) {
        scaled_ref_points_[i] = ref_pixel_uv[i] / scale;
    }

    // If sizeof ref_pixel_uv is not equal to cur_pixel_uv, view it as no prediction.
    if (scaled_ref_points_.size() != cur_pixel_uv.size()) {
        cur_pixel_uv = scaled_ref_points_;
    } else {
        for (uint32_t i = 0; i < cur_pixel_uv.size(); ++i) {
            cur_pixel_uv[i] /= scale;
        }
    }

    // If sizeof ref_pixel_uv is not equal to status, view it as all features haven't been tracked.
    if (scaled_ref_points_.size() != status.size()) {
        status.resize(scaled_ref_points_.size(), static_cast<uint8_t>(TrackStatus::kNotTracked));
    }

    // Track per level.
    for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
        const Image &ref_image = ref_pyramid.GetImageConst(level_idx);
        const Image &cur_image = cur_pyramid.GetImageConst(level_idx);

        TrackSingleLevel(ref_image, cur_image, scaled_ref_points_, cur_pixel_uv, status);

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

}
