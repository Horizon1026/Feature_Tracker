#ifndef _OPTICAL_FLOW_DATATYPE_H_
#define _OPTICAL_FLOW_DATATYPE_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "math_kinematics.h"

namespace OPTICAL_FLOW {

enum class TrackStatus : uint8_t {
    NOT_TRACKED = 0,
    TRACKED = 1,
    OUTSIDE = 2,
    LARGE_RESIDUAL = 3,
    NUM_ERROR = 4,
};

}

#endif
