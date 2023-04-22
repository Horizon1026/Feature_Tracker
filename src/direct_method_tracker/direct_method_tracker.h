#ifndef _DIRECT_METHOD_TRACKER_H_
#define _DIRECT_METHOD_TRACKER_H_

#include "datatype_basic.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "math_kinematics.h"
#include "feature_tracker.h"

#include "memory"

namespace FEATURE_TRACKER {

enum DirectMethodMethod : uint8_t {
    kInverse = 0,
    kDirect = 1,
    kFast = 2,
};

struct DirectMethodOptions {
    uint32_t kMaxTrackPointsNumber = 200;
    uint32_t kMaxIteration = 10;
    int32_t kPatchRowHalfSize = 6;
    int32_t kPatchColHalfSize = 6;
    float kMaxConvergeStep = 1e-2f;
    float kMaxConvergeResidual = 2.0f;
    DirectMethodMethod kMethod = kDirect;
};

class DirectMethod {

public:
    DirectMethod() = default;
    virtual ~DirectMethod() = default;
    DirectMethod(const DirectMethod &direct_method) = delete;

    bool TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                            const ImagePyramid &cur_pyramid,
                            const std::array<float, 4> &K,
                            const Quat ref_q_wc,
                            const Vec3 ref_p_wc,
                            const std::vector<Vec3> &p_w,
                            const std::vector<Vec2> &ref_pixel_uv,
                            std::vector<Vec2> &cur_pixel_uv,
                            Quat &cur_q_wc,
                            Vec3 &cur_p_wc);

    bool TrackMultipleLevel(const ImagePyramid &ref_pyramid,
                            const ImagePyramid &cur_pyramid,
                            const std::array<float, 4> &K,
                            const std::vector<Vec3> &p_c_in_ref,
                            const std::vector<Vec2> &ref_pixel_uv,
                            std::vector<Vec2> &cur_pixel_uv,
                            Quat &q_rc,
                            Vec3 &p_rc);

    DirectMethodOptions &options() { return options_; }

private:
    virtual bool TrackSingleLevel(const Image &ref_image,
                                  const Image &cur_image,
                                  const std::array<float, 4> &K,
                                  const std::vector<Vec3> &p_c_in_ref,
                                  const std::vector<Vec2> &ref_pixel_uv,
                                  std::vector<Vec2> &cur_pixel_uv,
                                  Quat &q_rc,
                                  Vec3 &p_rc);

    bool TrackAllFeaturesInverse(const Image &ref_image,
                                 const Image &cur_image,
                                 const std::array<float, 4> &K,
                                 const std::vector<Vec3> &p_c_in_ref,
                                 const std::vector<Vec2> &ref_pixel_uv,
                                 std::vector<Vec2> &cur_pixel_uv,
                                 Quat &q_rc,
                                 Vec3 &p_rc);

    bool TrackAllFeaturesDirect(const Image &ref_image,
                                const Image &cur_image,
                                const std::array<float, 4> &K,
                                const std::vector<Vec3> &p_c_in_ref,
                                const std::vector<Vec2> &ref_pixel_uv,
                                std::vector<Vec2> &cur_pixel_uv,
                                Quat &q_rc,
                                Vec3 &p_rc);

    bool TrackAllFeaturesFast(const Image &ref_image,
                              const Image &cur_image,
                              const std::array<float, 4> &K,
                              const std::vector<Vec3> &p_c_in_ref,
                              const std::vector<Vec2> &ref_pixel_uv,
                              std::vector<Vec2> &cur_pixel_uv,
                              Quat &q_rc,
                              Vec3 &p_rc);

private:
    DirectMethodOptions options_;

    // Scaled reference points pixel position for multi-level tracking.
    std::vector<Vec2> scaled_ref_points_ = {};

    // Points position in ref frame.
    std::vector<Vec3> p_c_in_ref_ = {};

    // Current frame pose in reference frame.
    Quat q_rc_ = Quat::Identity();
    Vec3 p_rc_ = Vec3::Zero();
};

}

#endif // end of _DIRECT_METHOD_TRACKER_H_
