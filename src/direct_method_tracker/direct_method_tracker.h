#ifndef _DIRECT_METHOD_TRACKER_H_
#define _DIRECT_METHOD_TRACKER_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "math_kinematics.h"
#include "feature_tracker.h"

namespace FEATURE_TRACKER {

enum DirectMethodMethod : uint8_t {
    INVERSE = 0,
    DIRECT = 1,
    FAST = 2,
};

struct OpticalFlowOptions {
    uint32_t kMaxTrackPointsNumber = 200;
    uint32_t kMaxIteration = 10;
    int32_t kPatchRowHalfSize = 6;
    int32_t kPatchColHalfSize = 6;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 2.0f;
    DirectMethodMethod kMethod = DIRECT;
};

}

#endif // end of _DIRECT_METHOD_TRACKER_H_
