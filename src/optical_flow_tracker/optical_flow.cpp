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
        status.resize(ref_pixel_uv.size(), static_cast<uint8_t>(TrackStatus::NOT_TRACKED));
    }

    // Prepare for tracking.
    PrepareForTracking();

    // Set predict and reference with scale.
    std::vector<Vec2> scaled_ref_points;
    scaled_ref_points.reserve(ref_pixel_uv.size());

    const int32_t scale = (2 << (ref_pyramid.level() - 1)) / 2;
    for (uint32_t i = 0; i < ref_pixel_uv.size(); ++i) {
        scaled_ref_points.emplace_back(ref_pixel_uv[i] / static_cast<float>(scale));
    }

    // If sizeof ref_pixel_uv is not equal to cur_pixel_uv, view it as no prediction.
    if (scaled_ref_points.size() != cur_pixel_uv.size()) {
        cur_pixel_uv = scaled_ref_points;
    } else {
        for (uint32_t i = 0; i < cur_pixel_uv.size(); ++i) {
            cur_pixel_uv[i] /= static_cast<float>(scale);
        }
    }

    // If sizeof ref_pixel_uv is not equal to status, view it as all features haven't been tracked.
    if (scaled_ref_points.size() != status.size()) {
        status.resize(scaled_ref_points.size(), static_cast<uint8_t>(TrackStatus::NOT_TRACKED));
    }

    // Track per level.
    for (int32_t level_idx = ref_pyramid.level() - 1; level_idx > -1; --level_idx) {
        const Image &ref_image = ref_pyramid.GetImageConst(level_idx);
        const Image &cur_image = cur_pyramid.GetImageConst(level_idx);

        TrackSingleLevel(ref_image, cur_image, scaled_ref_points, cur_pixel_uv, status);

        if (level_idx == 0) {
            break;
        }

        for (uint32_t i = 0; i < scaled_ref_points.size(); ++i) {
            scaled_ref_points[i] *= 2.0f;
            cur_pixel_uv[i] *= 2.0f;
        }
    }

    return true;
}

}
