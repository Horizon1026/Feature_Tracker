#ifndef _OPTICAL_FLOW_LSSD_KLT_H_
#define _OPTICAL_FLOW_LSSD_KLT_H_

#include "optical_flow.h"
#include <vector>

namespace FEATURE_TRACKER {

class OpticalFlowLssdKlt : public OpticalFlow {

public:
    OpticalFlowLssdKlt() : OpticalFlow() {}
    virtual ~OpticalFlowLssdKlt() = default;

private:
    virtual bool TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                                    const ImagePyramid &cur_pyramid,
                                    const std::vector<Vec2> &ref_pixel_uv,
                                    std::vector<Vec2> &cur_pixel_uv,
                                    std::vector<uint8_t> &status) override;
    virtual bool TrackSingleLevel(const GrayImage &ref_image,
                                  const GrayImage &cur_image,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  std::vector<uint8_t> &status) override;

    // Support for inverse/direct/fast method.
    void TrackOneFeature(const GrayImage &ref_image,
                         const GrayImage &cur_image,
                         const Vec2 &ref_pixel_uv,
                         Vec3 &se2_vector,
                         uint8_t &status);
    int32_t ConstructIncrementalFunction(const GrayImage &ref_image,
                                         const GrayImage &cur_image,
                                         const Vec2 &ref_pixel_uv,
                                         Vec3 &se2_vector,
                                         Mat3 &H,
                                         Vec3 &b);
    void TransformFeatureFromReferenceToCurrentImage(const Vec2 &ref_pixel_uv,
                                                     const Vec3 &se2_vector,
                                                     Vec2 &cur_pixel_uv);

    // Support for Sse method.

    // Support for Neon method.

};

}

#endif // end of _OPTICAL_FLOW_LSSD_KLT_H_
