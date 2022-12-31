#ifndef _OPTICAL_FLOW_DATATYPE_H_
#define _OPTICAL_FLOW_DATATYPE_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "math_kinematics.h"

namespace OPTICAL_FLOW {

typedef enum : uint8_t {
    TRACKED = 0,
    NOT_TRACKED,
    OUTSIDE,
    LARGE_RESIDUAL,
    NUM_ERROR,
} TrackStatus;

}

#endif
