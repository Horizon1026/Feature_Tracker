#ifndef _OPTICAL_FLOW_TRACKER_H_
#define _OPTICAL_FLOW_TRACKER_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "math_kinematics.h"

namespace FEATURE_TRACKER {

enum class TrackStatus : uint8_t {
    NOT_TRACKED = 0,
    TRACKED = 1,
    OUTSIDE = 2,
    LARGE_RESIDUAL = 3,
    NUM_ERROR = 4,
};

enum Method : uint8_t {
    LK_INVERSE = 0,
    LK_DIRECT = 1,
    LK_FAST = 2,
    KLT_INVERSE = 3,
    KLT_DIRECT = 4,
    KLT_FAST = 5,
};

struct Options {
    uint32_t kMaxTrackPointsNumber = 200;
    uint32_t kMaxIteration = 10;
    int32_t kPatchRowHalfSize = 6;
    int32_t kPatchColHalfSize = 6;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 2.0f;
    Method kMethod = LK_FAST;
};

class OpticalFlow {

public:
    OpticalFlow() = default;
    virtual ~OpticalFlow() = default;
    OpticalFlow(const OpticalFlow &optical_flow) = delete;

    bool TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                            const ImagePyramid &cur_pyramid,
                            const std::vector<Eigen::Vector2f> &ref_points,
                            std::vector<Eigen::Vector2f> &cur_points,
                            std::vector<uint8_t> &status);

    virtual bool TrackSingleLevel(const Image &ref_image,
                                  const Image &cur_image,
                                  const std::vector<Eigen::Vector2f> &ref_points,
                                  std::vector<Eigen::Vector2f> &cur_points,
                                  std::vector<uint8_t> &status) = 0;

    virtual bool PrepareForTracking() = 0;

    Options &options() { return options_; }

private:
    Options options_;

};

}

#endif
