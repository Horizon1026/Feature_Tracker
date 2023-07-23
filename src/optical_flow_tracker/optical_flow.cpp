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
        const GrayImage &ref_image = ref_pyramid.GetImageConst(level_idx);
        const GrayImage &cur_image = cur_pyramid.GetImageConst(level_idx);

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

uint32_t OpticalFlow::ExtractExtendPatchInReferenceImage(const GrayImage &ref_image,
                                                         const Vec2 &ref_pixel_uv,
                                                         int32_t ex_patch_rows,
                                                         int32_t ex_patch_cols,
                                                         std::vector<float> &ex_patch,
                                                         std::vector<bool> &ex_patch_pixel_valid) {
    // Compute the weight for linear interpolar.
    const float int_pixel_row = std::floor(ref_pixel_uv.y());
    const float int_pixel_col = std::floor(ref_pixel_uv.x());
    const float dec_pixel_row = ref_pixel_uv.y() - int_pixel_row;
    const float dec_pixel_col = ref_pixel_uv.x() - int_pixel_col;
    const float w_top_left = (1.0f - dec_pixel_row) * (1.0f - dec_pixel_col);
    const float w_top_right = (1.0f - dec_pixel_row) * dec_pixel_col;
    const float w_bottom_left = dec_pixel_row * (1.0f - dec_pixel_col);
    const float w_bottom_right = dec_pixel_row * dec_pixel_col;

    // Extract patch from reference image.
    const int32_t min_ref_pixel_row = static_cast<int32_t>(int_pixel_row) - ex_patch_rows / 2;
    const int32_t min_ref_pixel_col = static_cast<int32_t>(int_pixel_col) - ex_patch_cols / 2;
    const int32_t max_ref_pixel_row = min_ref_pixel_row + ex_patch_rows;
    const int32_t max_ref_pixel_col = min_ref_pixel_col + ex_patch_cols;

    if (min_ref_pixel_row < 0 || max_ref_pixel_row > ref_image.rows() - 2 ||
        min_ref_pixel_col < 0 || max_ref_pixel_col > ref_image.cols() - 2) {
        // If this patch is partly outside of reference image.
        uint32_t valid_pixel_cnt = 0;
        for (int32_t row = min_ref_pixel_row; row < max_ref_pixel_row; ++row) {
            for (int32_t col = min_ref_pixel_col; col < max_ref_pixel_col; ++col) {
                if (row < 0 || row > ref_image.rows() - 2 || col < 0 || col > ref_image.cols() - 2) {
                    ex_patch_pixel_valid.emplace_back(false);
                    ex_patch.emplace_back(0.0f);
                } else {
                    ex_patch_pixel_valid.emplace_back(true);
                    ex_patch.emplace_back(w_top_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col)) +
                                          w_top_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col + 1)) +
                                          w_bottom_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col)) +
                                          w_bottom_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col + 1)));
                    ++valid_pixel_cnt;
                }
            }
        }

        return valid_pixel_cnt;
    } else {
        // If this patch is totally inside of reference image.
        for (int32_t row = min_ref_pixel_row; row < max_ref_pixel_row; ++row) {
            for (int32_t col = min_ref_pixel_col; col < max_ref_pixel_col; ++col) {
                ex_patch_pixel_valid.emplace_back(true);
                ex_patch.emplace_back(w_top_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col)) +
                                      w_top_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row, col + 1)) +
                                      w_bottom_left * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col)) +
                                      w_bottom_right * static_cast<float>(ref_image.GetPixelValueNoCheck(row + 1, col + 1)));
            }
        }

        return ex_patch.size();
    }
}

bool OpticalFlow::PrepareForTracking() {
    patch_rows_ = (options_.kPatchRowHalfSize << 1) + 1;
    patch_cols_ = (options_.kPatchColHalfSize << 1) + 1;
    patch_size_ = patch_rows_ * patch_cols_;

    ex_patch_rows_ = patch_rows_ + 2;
    ex_patch_cols_ = patch_cols_ + 2;
    ex_patch_size_ = ex_patch_rows_ * ex_patch_cols_;

    ex_patch_.reserve(ex_patch_size_);
    ex_patch_pixel_valid_.reserve(ex_patch_size_);

    all_dx_.reserve(patch_size_);
    all_dy_.reserve(patch_size_);

    return true;
}

}
