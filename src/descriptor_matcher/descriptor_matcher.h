#ifndef _DESCRIPTOR_MATCHER_H_
#define _DESCRIPTOR_MATCHER_H_

#include "basic_type.h"
#include "datatype_image.h"
#include "feature_tracker.h"
#include "slam_basic_math.h"
#include "slam_operations.h"

#include "algorithm"
#include "cmath"
#include "limits"
#include "vector"

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
        struct {
            MatchMethod kMethod = MatchMethod::kForce;
            float kMinValidDescriptorSimilarity = 1.0f;
            float kMaxDescriptorDistanceRatioBetweenBestAndSecondBest = -1.0f;  // Disabled if not in (0, 1).
            bool kEnableCrossCheck = false;
        } matcher;
        struct {
            int32_t kMaxValidPredictRowDistance = 40;
            int32_t kMaxValidPredictColDistance = 40;
        } nearby;
        struct {
            float kSinkhornEpsilon = 0.1f;
            int32_t kSinkhornMaxIterations = 100;
            float kSinkhornConvergenceTolerance = 1e-6f;
        } sinkhorn;
        struct {
            int32_t kSearchCol = 150;
            int32_t kSearchRow = 40;
            int32_t kSampleStep = 4;
            int32_t kSearchStep = 4;
        } global_translation;
        struct {
            bool kEnabled = false;
            float kTolerance = 25.0f;
            uint32_t kMinMatches = 3;
        } displacement_consensus;
    };

public:
    DescriptorMatcher() = default;
    virtual ~DescriptorMatcher() = default;

    bool Match(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
               std::vector<int32_t> &index_pairs_in_cur);

    bool Match(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
               const std::vector<Vec2> &pixel_uv_pred_in_cur, const std::vector<Vec2> &pixel_uv_cur, std::vector<int32_t> &index_pairs_in_cur);

    // Estimate 2D global translation (col, row) between two images by coarse grid search.
    Vec2 EstimateGlobalTranslation(const GrayImage &ref_image, const GrayImage &cur_image) const;

    Options &options() { return options_; }
    const Options &options() const { return options_; }

private:
    // Shared by all MatchMethod implementations.
    virtual float ComputeSimilarity(const DescriptorType &descriptor_ref, const DescriptorType &descriptor_cur);

    // kForce: brute-force nearest-neighbor matching.
    bool ForceMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                    std::vector<int32_t> &index_pairs_in_cur);

    // kNearby: geometry-constrained local search matching.
    bool NearbyMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                     const std::vector<Vec2> &pixel_uv_pred_in_cur, const std::vector<Vec2> &pixel_uv_cur, std::vector<int32_t> &index_pairs_in_cur);

    // kHungarian: optimal assignment via Hungarian algorithm.
    bool HungarianMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                        std::vector<int32_t> &index_pairs_in_cur);
    bool SolveHungarianAssignment(const std::vector<std::vector<float>> &cost, std::vector<int32_t> &row_to_col) const;

    // kSinkhorn: soft optimal transport matching.
    bool SinkhornMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                       std::vector<int32_t> &index_pairs_in_cur);
    bool SolveSinkhornTransport(const std::vector<std::vector<float>> &kernel, int32_t max_iterations, float convergence_tolerance,
                                std::vector<std::vector<float>> &transport) const;

    // Match dispatch and geometry-Match post-processing.
    bool ExecuteMatchWithoutGeometry(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                     std::vector<int32_t> &index_pairs_in_cur);  // Route to kForce / kHungarian / kSinkhorn.
    static float ComputeMedian(std::vector<float> values);  // Helper for displacement-consensus filtering.
    bool FilterMatchesByDisplacementConsensus(const std::vector<Vec2> &ref_pixel_uv, const std::vector<Vec2> &pixel_uv_cur,
                                              std::vector<int32_t> &index_pairs_in_cur);  // Post-filter after geometry Match.

    // Shared by kForce and kNearby.
    bool PassDescriptorRatioTest(float best_similarity, float second_best_similarity) const;
    void ApplyCrossCheck(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                         std::vector<int32_t> &index_pairs_in_cur);

private:
    Options options_;
};

/* Class Descriptor Matcher Definition. */
template <typename DescriptorType>
float DescriptorMatcher<DescriptorType>::ComputeMedian(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    return values[mid];
}

template <typename DescriptorType>
Vec2 DescriptorMatcher<DescriptorType>::EstimateGlobalTranslation(const GrayImage &ref_image, const GrayImage &cur_image) const {
    // Brute-force translation search (coarse block matching on the full image).
    // For each candidate shift (disp_col, disp_row), warp ref pixels into cur and score how well
    // intensities agree. We use mean absolute difference (MAD, L1) instead of SSD (L2) for
    // robustness to outliers and illumination changes. The shift with the lowest MAD is returned.

    // Assumption: the dominant inter-frame motion is a global 2D translation (no rotation/scale).
    // kSearchCol/Row define the search window; kSearchStep subsamples candidates; kSampleStep
    // subsamples pixels inside each score for speed.
    float best_error = std::numeric_limits<float>::infinity();
    Vec2 best_shift = Vec2::Zero();
    const auto &search = options_.global_translation;

    for (int32_t disp_col = -search.kSearchCol; disp_col <= search.kSearchCol; disp_col += search.kSearchStep) {
        for (int32_t disp_row = -search.kSearchRow; disp_row <= search.kSearchRow; disp_row += search.kSearchStep) {
            float error = 0.0f;
            int32_t count = 0;
            for (int32_t row = 0; row < ref_image.rows(); row += search.kSampleStep) {
                for (int32_t col = 0; col < ref_image.cols(); col += search.kSampleStep) {
                    const int32_t cur_col = col + disp_col;
                    const int32_t cur_row = row + disp_row;
                    if (cur_col < 0 || cur_col >= cur_image.cols() || cur_row < 0 || cur_row >= cur_image.rows()) {
                        continue;
                    }
                    error += std::fabs(static_cast<float>(ref_image.GetPixelValueNoCheck(row, col)) -
                                       static_cast<float>(cur_image.GetPixelValueNoCheck(cur_row, cur_col)));
                    ++count;
                }
            }

            if (count == 0) {
                continue;
            }
            error /= static_cast<float>(count);  // Mean absolute difference for this candidate shift.
            if (error < best_error) {
                best_error = error;
                best_shift = Vec2(static_cast<float>(disp_col), static_cast<float>(disp_row));
            }
        }
    }

    // best_shift maps ref (col, row) -> cur: cur_uv = ref_uv + best_shift.
    return best_shift;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::FilterMatchesByDisplacementConsensus(const std::vector<Vec2> &ref_pixel_uv, const std::vector<Vec2> &pixel_uv_cur,
                                                                             std::vector<int32_t> &index_pairs_in_cur) {
    std::vector<float> dxs;
    std::vector<float> dys;
    dxs.reserve(index_pairs_in_cur.size());
    dys.reserve(index_pairs_in_cur.size());

    for (uint32_t i = 0; i < index_pairs_in_cur.size(); ++i) {
        const int32_t j = index_pairs_in_cur[i];
        if (j < 0 || static_cast<size_t>(j) >= pixel_uv_cur.size()) {
            continue;
        }
        dxs.push_back(pixel_uv_cur[static_cast<size_t>(j)].x() - ref_pixel_uv[i].x());
        dys.push_back(pixel_uv_cur[static_cast<size_t>(j)].y() - ref_pixel_uv[i].y());
    }

    if (!options_.displacement_consensus.kEnabled || dxs.size() < options_.displacement_consensus.kMinMatches) {
        return true;
    }

    const float median_dx = ComputeMedian(dxs);
    const float median_dy = ComputeMedian(dys);
    const float tolerance = options_.displacement_consensus.kTolerance;
    for (uint32_t i = 0; i < index_pairs_in_cur.size(); ++i) {
        const int32_t j = index_pairs_in_cur[i];
        if (j < 0 || static_cast<size_t>(j) >= pixel_uv_cur.size()) {
            continue;
        }
        const float dx = pixel_uv_cur[static_cast<size_t>(j)].x() - ref_pixel_uv[i].x();
        const float dy = pixel_uv_cur[static_cast<size_t>(j)].y() - ref_pixel_uv[i].y();
        if (std::fabs(dx - median_dx) > tolerance || std::fabs(dy - median_dy) > tolerance) {
            index_pairs_in_cur[i] = -1;
        }
    }

    return true;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::ExecuteMatchWithoutGeometry(const std::vector<DescriptorType> &descriptors_ref,
                                                                    const std::vector<DescriptorType> &descriptors_cur,
                                                                    std::vector<int32_t> &index_pairs_in_cur) {
    switch (options_.matcher.kMethod) {
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
                                              std::vector<int32_t> &index_pairs_in_cur) {
    RETURN_FALSE_IF(descriptors_cur.empty());
    if (descriptors_ref.size() != index_pairs_in_cur.size()) {
        index_pairs_in_cur.assign(descriptors_ref.size(), -1);
    }

    return ExecuteMatchWithoutGeometry(descriptors_ref, descriptors_cur, index_pairs_in_cur);
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

    const MatchMethod effective_method = options_.matcher.kMethod == MatchMethod::kNearby ? MatchMethod::kNearby : options_.matcher.kMethod;

    bool matched = false;
    if (effective_method == MatchMethod::kNearby) {
        matched = NearbyMatch(descriptors_ref, descriptors_cur, pixel_uv_pred_in_cur, pixel_uv_cur, index_pairs_in_cur);
    } else {
        matched = ExecuteMatchWithoutGeometry(descriptors_ref, descriptors_cur, index_pairs_in_cur);
    }

    if (options_.displacement_consensus.kEnabled) {
        FilterMatchesByDisplacementConsensus(pixel_uv_pred_in_cur, pixel_uv_cur, index_pairs_in_cur);
    }

    return matched;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::PassDescriptorRatioTest(float best_similarity, float second_best_similarity) const {
    const float ratio = options_.matcher.kMaxDescriptorDistanceRatioBetweenBestAndSecondBest;
    if (ratio <= 0.0f || ratio >= 1.0f) {
        return true;
    }
    if (second_best_similarity <= -std::numeric_limits<float>::infinity() + 1.0f) {
        return true;
    }

    const float best_distance = 1.0f - best_similarity;
    const float second_distance = 1.0f - second_best_similarity;
    if (second_distance < kZeroFloat) {
        return best_distance < kZeroFloat;
    }
    return best_distance < ratio * second_distance;
}

template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::ForceMatch(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                                   std::vector<int32_t> &index_pairs_in_cur) {
    const int32_t max_i = descriptors_ref.size();
    const int32_t max_j = descriptors_cur.size();
    for (int32_t i = 0; i < max_i; ++i) {
        index_pairs_in_cur[static_cast<size_t>(i)] = -1;
        float best_similarity = -std::numeric_limits<float>::infinity();
        float second_best_similarity = -std::numeric_limits<float>::infinity();
        int32_t best_j = -1;
        for (int32_t j = 0; j < max_j; ++j) {
            const float similarity = ComputeSimilarity(descriptors_ref[static_cast<size_t>(i)], descriptors_cur[static_cast<size_t>(j)]);
            if (similarity > best_similarity) {
                second_best_similarity = best_similarity;
                best_similarity = similarity;
                best_j = j;
            } else if (similarity > second_best_similarity) {
                second_best_similarity = similarity;
            }
        }

        if (best_j >= 0 && best_similarity >= options_.matcher.kMinValidDescriptorSimilarity &&
            PassDescriptorRatioTest(best_similarity, second_best_similarity)) {
            index_pairs_in_cur[static_cast<size_t>(i)] = best_j;
        }
    }

    ApplyCrossCheck(descriptors_ref, descriptors_cur, index_pairs_in_cur);
    return true;
}

template <typename DescriptorType>
void DescriptorMatcher<DescriptorType>::ApplyCrossCheck(const std::vector<DescriptorType> &descriptors_ref, const std::vector<DescriptorType> &descriptors_cur,
                                                        std::vector<int32_t> &index_pairs_in_cur) {
    if (!options_.matcher.kEnableCrossCheck || descriptors_ref.empty() || descriptors_cur.empty()) {
        return;
    }

    const int32_t num_ref = static_cast<int32_t>(descriptors_ref.size());
    const int32_t num_cur = static_cast<int32_t>(descriptors_cur.size());
    std::vector<int32_t> index_pairs_in_ref(static_cast<size_t>(num_cur), -1);

    for (int32_t j = 0; j < num_cur; ++j) {
        float best_similarity = -std::numeric_limits<float>::infinity();
        int32_t best_i = -1;
        for (int32_t i = 0; i < num_ref; ++i) {
            const float similarity = ComputeSimilarity(descriptors_ref[static_cast<size_t>(i)], descriptors_cur[static_cast<size_t>(j)]);
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_i = i;
            }
        }
        if (best_i >= 0 && best_similarity >= options_.matcher.kMinValidDescriptorSimilarity) {
            index_pairs_in_ref[static_cast<size_t>(j)] = best_i;
        }
    }

    for (int32_t i = 0; i < num_ref; ++i) {
        const int32_t j = index_pairs_in_cur[static_cast<size_t>(i)];
        if (j < 0 || j >= num_cur) {
            continue;
        }
        if (index_pairs_in_ref[static_cast<size_t>(j)] != i) {
            index_pairs_in_cur[static_cast<size_t>(i)] = -1;
        }
    }
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
            if (similarity >= options_.matcher.kMinValidDescriptorSimilarity) {
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
        if (similarity >= options_.matcher.kMinValidDescriptorSimilarity) {
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

    RETURN_FALSE_IF(options_.sinkhorn.kSinkhornEpsilon < kZeroFloat);
    RETURN_FALSE_IF(options_.sinkhorn.kSinkhornMaxIterations <= 0);

    std::vector<std::vector<float>> kernel(static_cast<size_t>(num_ref), std::vector<float>(static_cast<size_t>(num_cur), 0.0f));
    for (int32_t i = 0; i < num_ref; ++i) {
        for (int32_t j = 0; j < num_cur; ++j) {
            const float similarity = ComputeSimilarity(descriptors_ref[static_cast<size_t>(i)], descriptors_cur[static_cast<size_t>(j)]);
            if (similarity >= options_.matcher.kMinValidDescriptorSimilarity) {
                kernel[static_cast<size_t>(i)][static_cast<size_t>(j)] = std::exp(similarity / options_.sinkhorn.kSinkhornEpsilon);
            }
        }
    }

    std::vector<std::vector<float>> transport;
    RETURN_FALSE_IF_FALSE(SolveSinkhornTransport(kernel, options_.sinkhorn.kSinkhornMaxIterations, options_.sinkhorn.kSinkhornConvergenceTolerance, transport));

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
        if (similarity < options_.matcher.kMinValidDescriptorSimilarity) {
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
    const uint32_t max_i = descriptors_ref.size();
    const uint32_t max_j = descriptors_cur.size();
    for (uint32_t i = 0; i < max_i; ++i) {
        index_pairs_in_cur[i] = -1;
        float best_similarity = -std::numeric_limits<float>::infinity();
        float second_best_similarity = -std::numeric_limits<float>::infinity();
        int32_t best_j = -1;
        for (uint32_t j = 0; j < max_j; ++j) {
            if (std::fabs(pixel_uv_pred_in_cur[i].x() - pixel_uv_cur[j].x()) > options_.nearby.kMaxValidPredictColDistance ||
                std::fabs(pixel_uv_pred_in_cur[i].y() - pixel_uv_cur[j].y()) > options_.nearby.kMaxValidPredictRowDistance) {
                continue;
            }

            const float similarity = ComputeSimilarity(descriptors_ref[i], descriptors_cur[j]);
            if (similarity > best_similarity) {
                second_best_similarity = best_similarity;
                best_similarity = similarity;
                best_j = static_cast<int32_t>(j);
            } else if (similarity > second_best_similarity) {
                second_best_similarity = similarity;
            }

            BREAK_IF(similarity >= 1.0f - kZeroFloat);
        }

        if (best_j >= 0 && best_similarity >= options_.matcher.kMinValidDescriptorSimilarity &&
            PassDescriptorRatioTest(best_similarity, second_best_similarity)) {
            index_pairs_in_cur[i] = best_j;
        }
    }

    ApplyCrossCheck(descriptors_ref, descriptors_cur, index_pairs_in_cur);
    return true;
}

}  // namespace feature_tracker

#endif  // end of _DESCRIPTOR_MATCHER_H_
