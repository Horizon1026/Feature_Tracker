#ifndef _OPTICAL_FLOW_LK_H_
#define _OPTICAL_FLOW_LK_H_

#include "optical_flow_datatype.h"
#include <eigen3/Eigen/Eigen>
#include <vector>

namespace OPTICAL_FLOW {

typedef enum : uint8_t {
    TRACKED = 0,
    NOT_TRACKED,
    OUTSIDE,
    LARGE_RESIDUAL,
    NUM_ERROR,
} TrackStatus;

typedef struct {
    uint32_t kMaxTrackingPointsNumber = 200;
    uint32_t kMaxIteration = 10;
    int32_t kPatchRowHalfSize = 4;
    int32_t kPatchColHalfSize = 4;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 1e-2f;
} LkOptions;

class OpticalFlowLk {
public:
    explicit OpticalFlowLk() = default;
    virtual ~OpticalFlowLk() = default;

    bool TrackMultipleLevel(const ImagePyramid *ref_pyramid,
                            const ImagePyramid *cur_pyramid,
                            const std::vector<Eigen::Vector2f> &ref_points,
                            std::vector<Eigen::Vector2f> &cur_points,
                            std::vector<TrackStatus> &status);

    bool TrackSingleLevel(const Image *ref_image,
                          const Image *cur_image,
                          const std::vector<Eigen::Vector2f> &ref_points,
                          std::vector<Eigen::Vector2f> &cur_points,
                          std::vector<TrackStatus> &status);

    LkOptions &options() { return options_; }

private:
    LkOptions options_;
};

}

#endif
