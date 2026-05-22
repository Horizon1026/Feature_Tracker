#include "descriptor_matcher.h"
#include "slam_basic_math.h"

#include "limits"

namespace feature_tracker {

template bool DescriptorMatcher<std::vector<bool>>::SolveHungarianAssignment(const std::vector<std::vector<float>> &cost,
                                                                             std::vector<int32_t> &row_to_col) const;
template bool DescriptorMatcher<Vec>::SolveHungarianAssignment(const std::vector<std::vector<float>> &cost, std::vector<int32_t> &row_to_col) const;
template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::SolveHungarianAssignment(const std::vector<std::vector<float>> &cost, std::vector<int32_t> &row_to_col) const {
    const int32_t n = static_cast<int32_t>(cost.size());
    row_to_col.assign(n, -1);
    if (n == 0) {
        return true;
    }

    for (int32_t i = 0; i < n; ++i) {
        if (static_cast<int32_t>(cost[i].size()) != n) {
            return false;
        }
    }

    // Classic O(n^3) Hungarian algorithm (1-indexed internal state).
    std::vector<float> u(n + 1, 0.0f);
    std::vector<float> v(n + 1, 0.0f);
    std::vector<int32_t> p(n + 1, 0);
    std::vector<int32_t> way(n + 1, 0);

    for (int32_t i = 1; i <= n; ++i) {
        p[0] = i;
        int32_t j0 = 0;
        std::vector<float> minv(n + 1, std::numeric_limits<float>::infinity());
        std::vector<uint8_t> used(n + 1, 0);

        do {
            used[j0] = 1;
            const int32_t i0 = p[j0];
            float delta = std::numeric_limits<float>::infinity();
            int32_t j1 = 0;

            for (int32_t j = 1; j <= n; ++j) {
                if (used[j]) {
                    continue;
                }

                const float reduced_cost = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (reduced_cost < minv[j]) {
                    minv[j] = reduced_cost;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (int32_t j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int32_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    for (int32_t j = 1; j <= n; ++j) {
        if (p[j] != 0) {
            row_to_col[p[j] - 1] = j - 1;
        }
    }

    return true;
}

template bool DescriptorMatcher<std::vector<bool>>::SolveSinkhornTransport(const std::vector<std::vector<float>> &kernel, int32_t max_iterations,
                                                                           float convergence_tolerance, std::vector<std::vector<float>> &transport) const;
template bool DescriptorMatcher<Vec>::SolveSinkhornTransport(const std::vector<std::vector<float>> &kernel, int32_t max_iterations, float convergence_tolerance,
                                                             std::vector<std::vector<float>> &transport) const;
template <typename DescriptorType>
bool DescriptorMatcher<DescriptorType>::SolveSinkhornTransport(const std::vector<std::vector<float>> &kernel, int32_t max_iterations,
                                                               float convergence_tolerance, std::vector<std::vector<float>> &transport) const {
    const int32_t num_ref = static_cast<int32_t>(kernel.size());
    if (num_ref == 0) {
        transport.clear();
        return true;
    }

    const int32_t num_cur = static_cast<int32_t>(kernel[0].size());
    if (num_cur == 0) {
        return false;
    }

    for (int32_t i = 0; i < num_ref; ++i) {
        if (static_cast<int32_t>(kernel[static_cast<size_t>(i)].size()) != num_cur) {
            return false;
        }
    }

    transport.assign(static_cast<size_t>(num_ref), std::vector<float>(static_cast<size_t>(num_cur), 0.0f));

    const float inv_num_ref = 1.0f / static_cast<float>(num_ref);
    const float inv_num_cur = 1.0f / static_cast<float>(num_cur);
    std::vector<float> u(static_cast<size_t>(num_ref), 1.0f);
    std::vector<float> v(static_cast<size_t>(num_cur), 1.0f);
    std::vector<float> prev_u(static_cast<size_t>(num_ref), 0.0f);

    for (int32_t iter = 0; iter < max_iterations; ++iter) {
        prev_u = u;

        for (int32_t i = 0; i < num_ref; ++i) {
            float sum = 0.0f;
            for (int32_t j = 0; j < num_cur; ++j) {
                sum += kernel[static_cast<size_t>(i)][static_cast<size_t>(j)] * v[static_cast<size_t>(j)];
            }
            u[static_cast<size_t>(i)] = (sum > kZeroFloat) ? (inv_num_ref / sum) : 0.0f;
        }

        for (int32_t j = 0; j < num_cur; ++j) {
            float sum = 0.0f;
            for (int32_t i = 0; i < num_ref; ++i) {
                sum += kernel[static_cast<size_t>(i)][static_cast<size_t>(j)] * u[static_cast<size_t>(i)];
            }
            v[static_cast<size_t>(j)] = (sum > kZeroFloat) ? (inv_num_cur / sum) : 0.0f;
        }

        float max_u_change = 0.0f;
        for (int32_t i = 0; i < num_ref; ++i) {
            max_u_change = std::max(max_u_change, std::fabs(u[static_cast<size_t>(i)] - prev_u[static_cast<size_t>(i)]));
        }
        if (max_u_change < convergence_tolerance) {
            break;
        }
    }

    for (int32_t i = 0; i < num_ref; ++i) {
        for (int32_t j = 0; j < num_cur; ++j) {
            transport[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                u[static_cast<size_t>(i)] * kernel[static_cast<size_t>(i)][static_cast<size_t>(j)] * v[static_cast<size_t>(j)];
        }
    }

    return true;
}

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
