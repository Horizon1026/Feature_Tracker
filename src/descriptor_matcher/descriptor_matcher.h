#ifndef _DESCRIPTOR_MATCHER_H_
#define _DESCRIPTOR_MATCHER_H_

#include "datatype_basic.h"
#include "math_kinematics.h"

namespace FEATURE_TRACKER {

struct DescriptorMatcherOptions {
    int32_t kMaxValidSquareDistance = 40;
};

/* Class Descriptor Matcher Declaration. */
template <typename DescriptorType>
class DescriptorMatcher {

public:
    DescriptorMatcher() = default;
    virtual ~DescriptorMatcher() = default;

    bool ForceMatch(const std::vector<DescriptorType> &descriptors_ref,
                    const std::vector<DescriptorType> &descriptors_cur,
                    std::vector<int32_t> &index_pairs_in_cur);

    bool NearbyMatch(const std::vector<DescriptorType> &descriptors_ref,
                     const std::vector<DescriptorType> &descriptors_cur,
                     const std::vector<Vec2> &pixel_uv_ref,
                     const std::vector<Vec2> &pixel_uv_cur,
                     std::vector<int32_t> &index_pairs_in_cur);

    DescriptorMatcherOptions &options() { return options_; }

private:
    virtual int32_t ComputeDistance(const DescriptorType &descriptor_ref,
                                    const DescriptorType &descriptor_cur) = 0;

private:
    DescriptorMatcherOptions options_;

};

/* Class Descriptor Matcher Definition. */
template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::ForceMatch(const std::vector<DescriptorType> &descriptors_ref,
                                                   const std::vector<DescriptorType> &descriptors_cur,
                                                   std::vector<int32_t> &index_pairs_in_cur) {
    if (descriptors_cur.empty()) {
        return false;
    }

    if (descriptors_ref.size() != index_pairs_in_cur.size()) {
        index_pairs_in_cur.resize(descriptors_ref.size(), -1);
    }

    // For each descriptor in ref, find best pair in cur.
    const int32_t max_i = descriptors_ref.size();
    const int32_t max_j = descriptors_cur.size();
    for (int32_t i = 0; i < max_i; ++i) {
        int32_t min_distance = kMaxInt32;
        for (int32_t j = 0; j < max_j; ++j) {
            int32_t distance = ComputeDistance(descriptors_ref[i], descriptors_cur[j]);
            if (distance < min_distance) {
                min_distance = distance;
                index_pairs_in_cur[i] = j;
            }
        }
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::NearbyMatch(const std::vector<DescriptorType> &descriptors_ref,
                                                    const std::vector<DescriptorType> &descriptors_cur,
                                                    const std::vector<Vec2> &pixel_uv_ref,
                                                    const std::vector<Vec2> &pixel_uv_cur,
                                                    std::vector<int32_t> &index_pairs_in_cur) {
    if (descriptors_cur.empty()) {
        return false;
    }

    if (descriptors_ref.size() != index_pairs_in_cur.size()) {
        index_pairs_in_cur.resize(descriptors_ref.size(), -1);
    }

    // For each descriptor in ref, find best pair in cur.
    const uint32_t max_i = descriptors_ref.size();
    const uint32_t max_j = descriptors_cur.size();
    for (uint32_t i = 0; i < max_i; ++i) {
        int32_t min_distance = kMaxInt32;
        for (uint32_t j = 0; j < max_j; ++j) {
            if (std::fabs(pixel_uv_ref[i].x() - pixel_uv_cur[j].x()) > options_.kMaxValidSquareDistance ||
                std::fabs(pixel_uv_ref[i].y() - pixel_uv_cur[j].y()) > options_.kMaxValidSquareDistance) {
                continue;
            }

            int32_t distance = ComputeDistance(descriptors_ref[i], descriptors_cur[j]);
            if (distance < min_distance) {
                min_distance = distance;
                index_pairs_in_cur[i] = j;
            }
        }
    }

    return true;
}

}

#endif // end of _DESCRIPTOR_MATCHER_H_
