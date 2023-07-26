#ifndef _FEATURE_TRACKER_H_
#define _FEATURE_TRACKER_H_

#include "datatype_basic.h"

namespace FEATURE_TRACKER {

enum class TrackStatus : uint8_t {
    kNotTracked = 0,
    kTracked = 1,
    kLargeResidual = 2,
    kOutside = 3,
    kNumericError = 4,
};

}

#endif // end of _FEATURE_TRACKER_H_
