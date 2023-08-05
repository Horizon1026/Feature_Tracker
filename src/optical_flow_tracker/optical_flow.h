#ifndef _OPTICAL_FLOW_TRACKER_H_
#define _OPTICAL_FLOW_TRACKER_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "math_kinematics.h"
#include "feature_tracker.h"

namespace FEATURE_TRACKER {

enum class OpticalFlowMethod : uint8_t {
    kInverse = 0,
    kDirect = 1,
    kFast = 2,
    kSse = 3,
    kNeon = 4,
};

struct OpticalFlowOptions {
    uint32_t kMaxTrackPointsNumber = 500;
    uint32_t kMaxIteration = 15;
    uint32_t kMaxToleranceLargeStep = 3;
    int32_t kPatchRowHalfSize = 6;
    int32_t kPatchColHalfSize = 6;
    float kMaxConvergeStep = 4e-2f;
    OpticalFlowMethod kMethod = OpticalFlowMethod::kFast;
};

class OpticalFlow {

public:
    OpticalFlow() = default;
    virtual ~OpticalFlow() = default;

    bool TrackFeatures(const ImagePyramid &ref_pyramid,
                       const ImagePyramid &cur_pyramid,
                       const std::vector<Vec2> &ref_pixel_uv,
                       std::vector<Vec2> &cur_pixel_uv,
                       std::vector<uint8_t> &status);

    bool TrackFeatures(const GrayImage &ref_image,
                       const GrayImage &cur_image,
                       const std::vector<Vec2> &ref_pixel_uv,
                       std::vector<Vec2> &cur_pixel_uv,
                       std::vector<uint8_t> &status);

    // Support for all subclass's fast method.
    uint32_t ExtractExtendPatchInReferenceImage(const GrayImage &ref_image,
                                                const Vec2 &ref_pixel_uv,
                                                int32_t ex_ref_patch_rows,
                                                int32_t ex_ref_patch_cols,
                                                std::vector<float> &ex_ref_patch,
                                                std::vector<bool> &ex_ref_patch_pixel_valid);

    // Reference for member variables.
    OpticalFlowOptions &options() { return options_; }
    std::vector<float> &ex_ref_patch() { return ex_ref_patch_; }
    std::vector<bool> &ex_ref_patch_pixel_valid() { return ex_ref_patch_pixel_valid_; }
    std::vector<float> &all_dx_in_ref_patch() { return all_dx_in_ref_patch_; }
    std::vector<float> &all_dy_in_ref_patch() { return all_dy_in_ref_patch_; }
    int32_t &patch_rows() { return patch_rows_; }
    int32_t &patch_cols() { return patch_cols_; }
    int32_t &patch_size() { return patch_size_; }
    int32_t &ex_ref_patch_rows() { return ex_patch_rows_; }
    int32_t &ex_ref_patch_cols() { return ex_patch_cols_; }
    int32_t &ex_patch_size() { return ex_patch_size_; }

    // Const reference for member variables.
    const OpticalFlowOptions &options() const { return options_; }
    const std::vector<float> &ex_ref_patch() const { return ex_ref_patch_; }
    const std::vector<bool> &ex_ref_patch_pixel_valid() const { return ex_ref_patch_pixel_valid_; }
    const std::vector<float> &all_dx_in_ref_patch() const { return all_dx_in_ref_patch_; }
    const std::vector<float> &all_dy_in_ref_patch() const { return all_dy_in_ref_patch_; }
    const int32_t &patch_rows() const { return patch_rows_; }
    const int32_t &patch_cols() const { return patch_cols_; }
    const int32_t &patch_size() const { return patch_size_; }
    const int32_t &ex_ref_patch_rows() const { return ex_patch_rows_; }
    const int32_t &ex_ref_patch_cols() const { return ex_patch_cols_; }
    const int32_t &ex_patch_size() const { return ex_patch_size_; }

private:
    virtual bool TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                    const ImagePyramid &cur_pyramid,
                                    const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv,
                                    std::vector<uint8_t> &status) = 0;
    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) = 0;
    virtual bool PrepareForTracking();

private:
    // General options for optical flow trackers.
    OpticalFlowOptions options_;

    // Variables of reference patch supporting for fast method.
    std::vector<float> ex_ref_patch_;   // Extended patch with bound size 1.
    std::vector<bool> ex_ref_patch_pixel_valid_;
    std::vector<float> all_dx_in_ref_patch_;
    std::vector<float> all_dy_in_ref_patch_;

    // Variables of current patch supporting for fast method.
    std::vector<float> ex_cur_patch_;   // Extended patch with bound size 1.
    std::vector<bool> ex_cur_patch_pixel_valid_;
    std::vector<float> all_dx_in_cur_patch_;
    std::vector<float> all_dy_in_cur_patch_;

    // Parameters of ref and cur patch.
    int32_t patch_rows_ = 0;
    int32_t patch_cols_ = 0;
    int32_t patch_size_ = 0;
    int32_t ex_patch_rows_ = 0;
    int32_t ex_patch_cols_ = 0;
    int32_t ex_patch_size_ = 0;

};

}

#endif
