#ifndef _DESCRIPTOR_MATCHER_H_
#define _DESCRIPTOR_MATCHER_H_

#include "datatype_basic.h"
#include "math_kinematics.h"
#include "slam_operations.h"
#include "feature_tracker.h"

namespace FEATURE_TRACKER {

struct DescriptorMatcherOptions {
    int32_t kMaxValidPredictRowDistance = 40;
    int32_t kMaxValidPredictColDistance = 40;
    int32_t kMaxValidDescriptorDistance = 50;
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

    bool ForceMatch(const std::vector<DescriptorType> &descriptors_ref,
                    const std::vector<DescriptorType> &descriptors_cur,
                    const std::vector<Vec2> &pixel_uv_cur,
                    std::vector<Vec2> &matched_pixel_uv_cur,
                    std::vector<uint8_t> &status);

    bool NearbyMatch(const std::vector<DescriptorType> &descriptors_ref,
                     const std::vector<DescriptorType> &descriptors_cur,
                     const std::vector<Vec2> &pixel_uv_pred_in_cur,
                     const std::vector<Vec2> &pixel_uv_cur,
                     std::vector<int32_t> &index_pairs_in_cur);

    bool NearbyMatch(const std::vector<DescriptorType> &descriptors_ref,
                     const std::vector<DescriptorType> &descriptors_cur,
                     const std::vector<Vec2> &pixel_uv_pred_in_cur,
                     const std::vector<Vec2> &pixel_uv_cur,
                     std::vector<Vec2> &matched_pixel_uv_cur,
                     std::vector<uint8_t> &status);

    // Reference for member variables.
    DescriptorMatcherOptions &options() { return options_; }

    // Const reference for member variables.
    const DescriptorMatcherOptions &options() const { return options_; }

private:
    virtual int32_t ComputeDistance(const DescriptorType &descriptor_ref,
                                    const DescriptorType &descriptor_cur) = 0;

    bool FillMatchedPixelByPairIndices(const std::vector<int32_t> &index_pairs_in_cur,
                                       const std::vector<Vec2> &pixel_uv_cur,
                                       std::vector<Vec2> &matched_pixel_uv_cur,
                                       std::vector<uint8_t> &status);

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
            const int32_t distance = ComputeDistance(descriptors_ref[i], descriptors_cur[j]);
            if (distance < min_distance && distance < options_.kMaxValidDescriptorDistance) {
                min_distance = distance;
                index_pairs_in_cur[i] = j;
            }
        }
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::ForceMatch(const std::vector<DescriptorType> &descriptors_ref,
                                                   const std::vector<DescriptorType> &descriptors_cur,
                                                   const std::vector<Vec2> &pixel_uv_cur,
                                                   std::vector<Vec2> &matched_pixel_uv_cur,
                                                   std::vector<uint8_t> &status) {
    std::vector<int32_t> index_pairs_in_cur;
    RETURN_FALSE_IF_FALSE(ForceMatch(descriptors_ref, descriptors_cur, index_pairs_in_cur));
	return FillMatchedPixelByPairIndices(index_pairs_in_cur, pixel_uv_cur, matched_pixel_uv_cur, status);
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::NearbyMatch(const std::vector<DescriptorType> &descriptors_ref,
                                                    const std::vector<DescriptorType> &descriptors_cur,
                                                    const std::vector<Vec2> &pixel_uv_pred_in_cur,
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
            if (std::fabs(pixel_uv_pred_in_cur[i].x() - pixel_uv_cur[j].x()) > options_.kMaxValidPredictColDistance ||
                std::fabs(pixel_uv_pred_in_cur[i].y() - pixel_uv_cur[j].y()) > options_.kMaxValidPredictRowDistance) {
                continue;
            }

            const int32_t distance = ComputeDistance(descriptors_ref[i], descriptors_cur[j]);
            if (distance < min_distance && distance < options_.kMaxValidDescriptorDistance) {
                min_distance = distance;
                index_pairs_in_cur[i] = j;
            }

            if (distance == 0) {
                break;
            }
        }
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::NearbyMatch(const std::vector<DescriptorType> &descriptors_ref,
                                                    const std::vector<DescriptorType> &descriptors_cur,
                                                    const std::vector<Vec2> &pixel_uv_pred_in_cur,
                                                    const std::vector<Vec2> &pixel_uv_cur,
                                                    std::vector<Vec2> &matched_pixel_uv_cur,
                                                    std::vector<uint8_t> &status) {
    std::vector<int32_t> index_pairs_in_cur;
    RETURN_FALSE_IF_FALSE(NearbyMatch(descriptors_ref, descriptors_cur, pixel_uv_pred_in_cur, pixel_uv_cur, index_pairs_in_cur));
	return FillMatchedPixelByPairIndices(index_pairs_in_cur, pixel_uv_cur, matched_pixel_uv_cur, status);
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::FillMatchedPixelByPairIndices(const std::vector<int32_t> &index_pairs_in_cur,
                                                                      const std::vector<Vec2> &pixel_uv_cur,
                                                                      std::vector<Vec2> &matched_pixel_uv_cur,
                                                                      std::vector<uint8_t> &status) {
    if (pixel_uv_cur.size() != status.size()) {
        status.resize(pixel_uv_cur.size(), static_cast<uint8_t>(TrackStatus::kNotTracked));
    }
    matched_pixel_uv_cur = pixel_uv_cur;

    for (uint32_t i = 0; i < matched_pixel_uv_cur.size(); ++i) {
        if (index_pairs_in_cur[i] > 0) {
            matched_pixel_uv_cur[i] = pixel_uv_cur[index_pairs_in_cur[i]];
            status[i] = static_cast<uint8_t>(TrackStatus::kTracked);
        } else {
            status[i] = static_cast<uint8_t>(TrackStatus::kLargeResidual);
        }
    }

    return true;
}

}

#endif // end of _DESCRIPTOR_MATCHER_H_
