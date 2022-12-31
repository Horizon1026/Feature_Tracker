#ifndef _OPTICAL_FLOW_KLT_H_
#define _OPTICAL_FLOW_KLT_H_

#include "optical_flow_datatype.h"
#include <eigen3/Eigen/Eigen>
#include <vector>

namespace OPTICAL_FLOW {

enum KltMethod : uint8_t {
    KLT_INVERSE = 0,
    KLT_DIRECT,
};

typedef struct {
    uint32_t kMaxTrackPointsNumber = 200;
    uint32_t kMaxIteration = 10;
    int32_t kPatchRowHalfSize = 6;
    int32_t kPatchColHalfSize = 6;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 1e-2f;
    KltMethod kMethod = KLT_INVERSE;
} KltOptions;

class OpticalFlowKlt {
public:
    explicit OpticalFlowKlt() = default;
    virtual ~OpticalFlowKlt() = default;

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

    KltOptions &options() { return options_; }

private:
    void TrackOneFeatureInverse(const Image *ref_image,
                                const Image *cur_image,
                                const Eigen::Vector2f &ref_points,
                                Eigen::Vector2f &cur_points,
                                TrackStatus &status);

    void TrackOneFeatureDirect(const Image *ref_image,
                               const Image *cur_image,
                               const Eigen::Vector2f &ref_points,
                               Eigen::Vector2f &cur_points,
                               TrackStatus &status);

private:
    KltOptions options_;
};

}

#endif
