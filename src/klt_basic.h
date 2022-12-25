#ifndef _KLT_BASIC_H_
#define _KLT_BASIC_H_

#include "klt_datatype.h"
#include <vector>
#include <eigen3/Eigen/Eigen>

namespace KLT_TRACKER {

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
    int32_t kPatchHalfSize = 4;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 1e-2f;
} KltBasicOptions;

class KltBasic {
public:
    explicit KltBasic() = default;
    virtual ~KltBasic() = default;

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

    KltBasicOptions &options() { return options_; }

private:
    KltBasicOptions options_;
};

}

#endif
