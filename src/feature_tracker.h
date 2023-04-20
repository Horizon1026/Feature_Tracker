#ifndef _FEATURE_TRACKER_H_
#define _FEATURE_TRACKER_H_

#include "datatype_basic.h"

namespace FEATURE_TRACKER {

enum class TrackStatus : uint8_t {
    NOT_TRACKED = 0,
    TRACKED = 1,
    OUTSIDE = 2,
    LARGE_RESIDUAL = 3,
    NUM_ERROR = 4,
};

}

#endif // end of _FEATURE_TRACKER_H_
