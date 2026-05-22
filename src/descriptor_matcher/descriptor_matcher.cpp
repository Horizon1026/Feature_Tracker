#include "descriptor_matcher.h"
#include "slam_basic_math.h"

namespace feature_tracker {

template <>
float DescriptorMatcher<std::vector<bool>>::ComputeSimilarity(const std::vector<bool> &descriptor_ref, const std::vector<bool> &descriptor_cur) {
    if (descriptor_ref.empty() || descriptor_cur.empty() || descriptor_ref.size() != descriptor_cur.size()) {
        return -1.0f;
    }

    float dot = 0.0f;
    float squared_norm_ref = 0.0f;
    float squared_norm_cur = 0.0f;
    for (uint32_t i = 0; i < descriptor_ref.size(); ++i) {
        const float value_ref = descriptor_ref[i] ? 1.0f : -1.0f;
        const float value_cur = descriptor_cur[i] ? 1.0f : -1.0f;
        dot += value_ref * value_cur;
        squared_norm_ref += value_ref * value_ref;
        squared_norm_cur += value_cur * value_cur;
    }

    if (squared_norm_ref < kZeroFloat || squared_norm_cur < kZeroFloat) {
        return -1.0f;
    }
    return dot / std::sqrt(squared_norm_ref * squared_norm_cur);
}

template <>
float DescriptorMatcher<Vec>::ComputeSimilarity(const Vec &descriptor_ref, const Vec &descriptor_cur) {
    if (descriptor_ref.size() == 0 || descriptor_ref.size() != descriptor_cur.size()) {
        return -1.0f;
    }

    const float norm_ref = descriptor_ref.norm();
    const float norm_cur = descriptor_cur.norm();
    if (norm_ref < kZeroFloat || norm_cur < kZeroFloat) {
        return -1.0f;
    }
    return descriptor_ref.dot(descriptor_cur) / (norm_ref * norm_cur);
}

}  // namespace feature_tracker
