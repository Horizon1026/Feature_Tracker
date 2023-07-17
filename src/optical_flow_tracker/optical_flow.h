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
    uint32_t kMaxTrackPointsNumber = 200;
    uint32_t kMaxIteration = 10;
    int32_t kPatchRowHalfSize = 6;
    int32_t kPatchColHalfSize = 6;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 2.0f;
    OpticalFlowMethod kMethod = OpticalFlowMethod::kFast;
};

class OpticalFlow {

public:
    OpticalFlow() = default;
    virtual ~OpticalFlow() = default;

    bool TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                            const ImagePyramid &cur_pyramid,
                            const std::vector<Vec2> &ref_pixel_uv,
                            std::vector<Vec2> &cur_pixel_uv,
                            std::vector<uint8_t> &status);

    // Reference for member variables.
    OpticalFlowOptions &options() { return options_; }

    // Const reference for member variables.
    const OpticalFlowOptions &options() const { return options_; }

private:
    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) = 0;

    virtual bool PrepareForTracking() = 0;

private:
    OpticalFlowOptions options_;

    // Scaled reference points pixel position for multi-level tracking.
    std::vector<Vec2> scaled_ref_points_ = {};

};

}

#endif
