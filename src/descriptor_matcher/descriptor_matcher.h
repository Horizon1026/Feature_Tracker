#ifndef _DESCRIPTOR_MATCHER_H_
#define _DESCRIPTOR_MATCHER_H_

#include "basic_type.h"
#include "feature_tracker.h"
#include "slam_basic_math.h"
#include "slam_operations.h"

#include "cmath"
#include "algorithm"

namespace feature_tracker {

/* Class Descriptor Matcher Declaration. */
template <typename DescriptorType>
class DescriptorMatcher {

public:
    enum class MatchMethod : uint8_t {
        kForce = 0,
        kNearby = 1,
        kHungarian = 2,
        kSinkhorn = 3,
    };
    struct Options {
        MatchMethod kMethod = MatchMethod::kForce;
        // Support for nearby match.
        int32_t kMaxValidPredictRowDistance = 40;
        int32_t kMaxValidPredictColDistance = 40;
        // Minimum cosine similarity for a valid match. Default 1.0 rejects all matches.
        float kMinValidDescriptorSimilarity = 1.0f;
        // Support for Sinkhorn match.
        float kSinkhornEpsilon = 0.1f;
        int32_t kSinkhornMaxIterations = 100;
        float kSinkhornConvergenceTolerance = 1e-6f;
    };

public:
    DescriptorMatcher() = default;
    virtual ~DescriptorMatcher() = default;

    bool Match(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
               std::vector<int32_t> &index_pairs_in_cur);

    bool Match(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
               const std::vector<Vec2> &pixel_uv_pred_in_cur, const std::vector<Vec2> &pixel_uv_cur, std::vector<int32_t> &index_pairs_in_cur);

    bool FillMatchedPixelByPairIndices(const std::vector<int32_t> &index_pairs_in_cur, const std::vector<Vec2> &pixel_uv_cur,
                                       std::vector<Vec2> &matched_pixel_uv_cur, std::vector<uint8_t> &status);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    virtual float ComputeSimilarity(const DescriptorType &descriptor_ref, const DescriptorType &descriptor_cur);

    bool ForceMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                    std::vector<int32_t> &index_pairs_in_cur);
    bool HungarianMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                        std::vector<int32_t> &index_pairs_in_cur);
    bool SinkhornMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                       std::vector<int32_t> &index_pairs_in_cur);
    // Minimum-cost perfect matching on a square cost matrix (rows to columns).
    // row_to_col[i] = j means row i is assigned to column j.
    bool SolveHungarianAssignment(const std::vector<std::vector<float>> &cost, std::vector<int32_t> &row_to_col) const;
    // Sinkhorn-Knopp transport on a non-negative kernel matrix (rows to columns).
    bool SolveSinkhornTransport(const std::vector<std::vector<float>> &kernel, int32_t max_iterations, float convergence_tolerance,
                                std::vector<std::vector<float>> &transport) const;
    bool NearbyMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                     const std::vector<Vec2> &pixel_uv_pred_in_cur, const std::vector<Vec2> &pixel_uv_cur, std::vector<int32_t> &index_pairs_in_cur);

private:
    Options options_;
};

/* Class Descriptor Matcher Definition. */
template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::Match(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                              std::vector<int32_t> &index_pairs_in_cur) {
    RETURN_FALSE_IF(descriptors_cur.empty());
    if (descriptors_ref.size() != index_pairs_in_cur.size()) {
        index_pairs_in_cur.assign(descriptors_ref.size(), -1);
    }

    switch (options_.kMethod) {
        case MatchMethod::kForce:
            return ForceMatch(descriptors_ref, descriptors_cur, index_pairs_in_cur);
        case MatchMethod::kHungarian:
            return HungarianMatch(descriptors_ref, descriptors_cur, index_pairs_in_cur);
        case MatchMethod::kSinkhorn:
            return SinkhornMatch(descriptors_ref, descriptors_cur, index_pairs_in_cur);
        default:
            return false;
    }
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::Match(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                              const std::vector<Vec2> &pixel_uv_pred_in_cur, const std::vector<Vec2> &pixel_uv_cur,
                                              std::vector<int32_t> &index_pairs_in_cur) {
    RETURN_FALSE_IF(descriptors_cur.empty());
    RETURN_FALSE_IF(descriptors_ref.size() != pixel_uv_pred_in_cur.size());
    RETURN_FALSE_IF(descriptors_cur.size() != pixel_uv_cur.size());
    if (descriptors_ref.size() != index_pairs_in_cur.size()) {
        index_pairs_in_cur.assign(descriptors_ref.size(), -1);
    }

    switch (options_.kMethod) {
        case MatchMethod::kForce:
            return ForceMatch(descriptors_ref, descriptors_cur, index_pairs_in_cur);
        case MatchMethod::kHungarian:
            return HungarianMatch(descriptors_ref, descriptors_cur, index_pairs_in_cur);
        case MatchMethod::kSinkhorn:
            return SinkhornMatch(descriptors_ref, descriptors_cur, index_pairs_in_cur);
        case MatchMethod::kNearby:
            return NearbyMatch(descriptors_ref, descriptors_cur, pixel_uv_pred_in_cur, pixel_uv_cur, index_pairs_in_cur);
        default:
            return false;
    }
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::ForceMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                                   std::vector<int32_t> &index_pairs_in_cur) {
    // For each descriptor in ref, find best pair in cur.
    const int32_t max_i = descriptors_ref.size();
    const int32_t max_j = descriptors_cur.size();
    for (int32_t i = 0; i < max_i; ++i) {
        float best_similarity = -std::numeric_limits<float>::infinity();
        for (int32_t j = 0; j < max_j; ++j) {
            const float similarity = ComputeSimilarity(descriptors_ref[i], descriptors_cur[j]);
            if (similarity >= options_.kMinValidDescriptorSimilarity && similarity > best_similarity) {
                best_similarity = similarity;
                index_pairs_in_cur[i] = j;
            }
        }
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::HungarianMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                                       std::vector<int32_t> &index_pairs_in_cur) {
    const int32_t num_ref = static_cast<int32_t>(descriptors_ref.size());
    const int32_t num_cur = static_cast<int32_t>(descriptors_cur.size());
    if (num_ref == 0) {
        return true;
    }

    const int32_t matrix_size = std::max(num_ref, num_cur);
    constexpr float kInvalidPairCost = 1e6f;
    std::vector<std::vector<float>> cost(static_cast<size_t>(matrix_size), std::vector<float>(static_cast<size_t>(matrix_size), kInvalidPairCost));

    for (int32_t i = 0; i < num_ref; ++i) {
        for (int32_t j = 0; j < num_cur; ++j) {
            const float similarity = ComputeSimilarity(descriptors_ref[i], descriptors_cur[j]);
            if (similarity >= options_.kMinValidDescriptorSimilarity) {
                cost[static_cast<size_t>(i)][static_cast<size_t>(j)] = -similarity;
            }
        }
    }

    std::vector<int32_t> assignment;
    RETURN_FALSE_IF_FALSE(SolveHungarianAssignment(cost, assignment));

    for (int32_t i = 0; i < num_ref; ++i) {
        index_pairs_in_cur[static_cast<size_t>(i)] = -1;
        const int32_t j = assignment[static_cast<size_t>(i)];
        if (j < 0 || j >= num_cur) {
            continue;
        }

        const float similarity = ComputeSimilarity(descriptors_ref[static_cast<size_t>(i)], descriptors_cur[static_cast<size_t>(j)]);
        if (similarity >= options_.kMinValidDescriptorSimilarity) {
            index_pairs_in_cur[static_cast<size_t>(i)] = j;
        }
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::SinkhornMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                                      std::vector<int32_t> &index_pairs_in_cur) {
    const int32_t num_ref = static_cast<int32_t>(descriptors_ref.size());
    const int32_t num_cur = static_cast<int32_t>(descriptors_cur.size());
    if (num_ref == 0) {
        return true;
    }

    RETURN_FALSE_IF(options_.kSinkhornEpsilon < kZeroFloat);
    RETURN_FALSE_IF(options_.kSinkhornMaxIterations <= 0);

    std::vector<std::vector<float>> kernel(static_cast<size_t>(num_ref), std::vector<float>(static_cast<size_t>(num_cur), 0.0f));
    for (int32_t i = 0; i < num_ref; ++i) {
        for (int32_t j = 0; j < num_cur; ++j) {
            const float similarity = ComputeSimilarity(descriptors_ref[static_cast<size_t>(i)], descriptors_cur[static_cast<size_t>(j)]);
            if (similarity >= options_.kMinValidDescriptorSimilarity) {
                kernel[static_cast<size_t>(i)][static_cast<size_t>(j)] = std::exp(similarity / options_.kSinkhornEpsilon);
            }
        }
    }

    std::vector<std::vector<float>> transport;
    RETURN_FALSE_IF_FALSE(SolveSinkhornTransport(kernel, options_.kSinkhornMaxIterations, options_.kSinkhornConvergenceTolerance, transport));

    struct Candidate {
        int32_t ref_id = -1;
        int32_t cur_id = -1;
        float mass = 0.0f;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(num_ref) * static_cast<size_t>(num_cur));
    for (int32_t i = 0; i < num_ref; ++i) {
        for (int32_t j = 0; j < num_cur; ++j) {
            const float mass = transport[static_cast<size_t>(i)][static_cast<size_t>(j)];
            if (mass > kZeroFloat) {
                candidates.push_back({i, j, mass});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate &lhs, const Candidate &rhs) {
        return lhs.mass > rhs.mass;
    });

    std::vector<uint8_t> ref_assigned(static_cast<size_t>(num_ref), 0);
    std::vector<uint8_t> cur_assigned(static_cast<size_t>(num_cur), 0);
    for (int32_t i = 0; i < num_ref; ++i) {
        index_pairs_in_cur[static_cast<size_t>(i)] = -1;
    }

    for (const Candidate &candidate: candidates) {
        if (ref_assigned[static_cast<size_t>(candidate.ref_id)] || cur_assigned[static_cast<size_t>(candidate.cur_id)]) {
            continue;
        }

        const float similarity =
            ComputeSimilarity(descriptors_ref[static_cast<size_t>(candidate.ref_id)], descriptors_cur[static_cast<size_t>(candidate.cur_id)]);
        if (similarity < options_.kMinValidDescriptorSimilarity) {
            continue;
        }

        index_pairs_in_cur[static_cast<size_t>(candidate.ref_id)] = candidate.cur_id;
        ref_assigned[static_cast<size_t>(candidate.ref_id)] = 1;
        cur_assigned[static_cast<size_t>(candidate.cur_id)] = 1;
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::NearbyMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                                    const std::vector<Vec2> &pixel_uv_pred_in_cur, const std::vector<Vec2> &pixel_uv_cur,
                                                    std::vector<int32_t> &index_pairs_in_cur) {
    // For each descriptor in ref, find best pair in cur.
    const uint32_t max_i = descriptors_ref.size();
    const uint32_t max_j = descriptors_cur.size();
    for (uint32_t i = 0; i < max_i; ++i) {
        float best_similarity = -std::numeric_limits<float>::infinity();
        for (uint32_t j = 0; j < max_j; ++j) {
            if (std::fabs(pixel_uv_pred_in_cur[i].x() - pixel_uv_cur[j].x()) > options_.kMaxValidPredictColDistance ||
                std::fabs(pixel_uv_pred_in_cur[i].y() - pixel_uv_cur[j].y()) > options_.kMaxValidPredictRowDistance) {
                continue;
            }

            const float similarity = ComputeSimilarity(descriptors_ref[i], descriptors_cur[j]);
            if (similarity >= options_.kMinValidDescriptorSimilarity && similarity > best_similarity) {
                best_similarity = similarity;
                index_pairs_in_cur[i] = j;
            }

            BREAK_IF(similarity >= 1.0f - kZeroFloat);
        }
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::FillMatchedPixelByPairIndices(const std::vector<int32_t> &index_pairs_in_cur, const std::vector<Vec2> &pixel_uv_cur,
                                                                      std::vector<Vec2> &matched_pixel_uv_cur, std::vector<uint8_t> &status) {
    if (index_pairs_in_cur.size() != status.size()) {
        status.assign(index_pairs_in_cur.size(), static_cast<uint8_t>(TrackStatus::kNotTracked));
    }

    matched_pixel_uv_cur.resize(index_pairs_in_cur.size());
    for (uint32_t ref_id = 0; ref_id < index_pairs_in_cur.size(); ++ref_id) {
        // Do not repeatly match features that has been tracking failed.
        CONTINUE_IF(status[ref_id] > static_cast<uint8_t>(TrackStatus::kTracked));

        const int32_t cur_id = index_pairs_in_cur[ref_id];
        if (cur_id >= 0 && static_cast<size_t>(cur_id) < pixel_uv_cur.size()) {
            matched_pixel_uv_cur[ref_id] = pixel_uv_cur[cur_id];
            status[ref_id] = static_cast<uint8_t>(TrackStatus::kTracked);
        } else {
            status[ref_id] = static_cast<uint8_t>(TrackStatus::kLargeResidual);
        }
    }

    return true;
}

}  // namespace feature_tracker

#endif  // end of _DESCRIPTOR_MATCHER_H_
