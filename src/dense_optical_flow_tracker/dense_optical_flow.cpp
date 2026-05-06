#include "dense_optical_flow.h"
#include "slam_operations.h"
#include "slam_log_reporter.h"

namespace feature_tracker {

bool DenseOpticalFlow::Track(const GrayImage &ref_image, const GrayImage &cur_image, std::array<Mat, 2> &flow_rc) {
    // Return false if input images are invalid
    RETURN_FALSE_IF(ref_image.data() == nullptr);
    RETURN_FALSE_IF(cur_image.data() == nullptr);
    // Initialize Gaussian kernel for weighted computation
    RETURN_FALSE_IF_FALSE(InitializeGaussianKernel());
    // Compute Gaussian weighted second moment matrices for both reference and current images
    RETURN_FALSE_IF_FALSE(ComputeGaussianWeightedSecondMomentMatrix(ref_image, 0));
    RETURN_FALSE_IF_FALSE(ComputeGaussianWeightedSecondMomentMatrix(cur_image, 1));

    // Initialize flow field matrix with zero values if size mismatch
    if (flow_rc[0].rows() != ref_image.rows() || flow_rc[0].cols() != ref_image.cols()) {
        flow_rc[0].setZero(ref_image.rows(), ref_image.cols());
    }
    if (flow_rc[1].rows() != ref_image.rows() || flow_rc[1].cols() != ref_image.cols()) {
        flow_rc[1].setZero(ref_image.rows(), ref_image.cols());
    }
    // Compute optical flow for each pixel individually
    for (int32_t row = 0; row < ref_image.rows(); ++row) {
        for (int32_t col = 0; col < ref_image.cols(); ++col) {
            ComputeFlowByPixel(row, col, flow_rc);
        }
    }
    // Smooth flow field to remove outliers and maintain spatial consistency
    SmoothFlow(flow_rc);
    return true;
}

bool DenseOpticalFlow::Track(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, std::array<Mat, 2> &flow_rc) {
    // Validate input pyramid data and level consistency
    RETURN_FALSE_IF(ref_pyramid.data() == nullptr);
    RETURN_FALSE_IF(cur_pyramid.data() == nullptr);
    RETURN_FALSE_IF(ref_pyramid.level() != cur_pyramid.level());

    // Initialize temporary flow storage for top pyramid level
    std::array<Mat, 2> temp_flow;
    const int32_t max_level = ref_pyramid.level() - 1;
    const GrayImage &top_img = ref_pyramid.GetImageConst(max_level);
    temp_flow[0].setZero(top_img.rows(), top_img.cols());
    temp_flow[1].setZero(top_img.rows(), top_img.cols());

    // Coarse-to-fine optical flow estimation
    for (int32_t level = max_level; level >= 0; --level) {
        const GrayImage &ref_img = ref_pyramid.GetImageConst(level);
        const GrayImage &cur_img = cur_pyramid.GetImageConst(level);
        // Compute flow at current pyramid level
        Track(ref_img, cur_img, temp_flow);

        // Copy final flow to output at the finest level (level 0)
        if (level == 0) {
            flow_rc[0] = std::move(temp_flow[0]);
            flow_rc[1] = std::move(temp_flow[1]);
            break;
        }

        // Upsample flow field to higher resolution level (bilinear interpolation)
        const GrayImage &next_img = ref_pyramid.GetImageConst(level - 1);
        std::array<Mat, 2> up_flow;
        up_flow[0].resize(next_img.rows(), next_img.cols());
        up_flow[1].resize(next_img.rows(), next_img.cols());

        for (int32_t r = 0; r < next_img.rows(); ++r) {
            for (int32_t c = 0; c < next_img.cols(); ++c) {
                // Scale coordinates for upsampling (2x resolution increase)
                const float fr = r * 0.5f;
                const float fc = c * 0.5f;
                // Bilinear interpolation and scale flow magnitude by 2
                up_flow[0](r, c) = slam_utility::Utility::Interpolate(temp_flow[0], fr, fc) * 2.0f;
                up_flow[1](r, c) = slam_utility::Utility::Interpolate(temp_flow[1], fr, fc) * 2.0f;
            }
        }

        // Update temporary flow with upsampled values for next iteration
        temp_flow[0] = std::move(up_flow[0]);
        temp_flow[1] = std::move(up_flow[1]);
    }

    return true;
}

bool DenseOpticalFlow::InitializeGaussianKernel() {
    RETURN_FALSE_IF(options_.kHalfPatchSize < 0);

    const int32_t center = options_.kHalfPatchSize;
    const int32_t kernel_size = 2 * center + 1;
    gaussian_kernel_.kernel_mat.setZero(kernel_size, kernel_size);

    // Handle 1x1 kernel (single pixel).
    if (center == 0) {
        gaussian_kernel_.kernel_mat(center, center) = 1.0f;
        return true;
    }

    // Fixed sigma value optimized for 5x5 patch (standard Farneback parameter).
    const float sigma = 1.0f;
    const float sigma2 = sigma * sigma;
    float sum = 0.0f;

    // Compute unnormalized Gaussian weights.
    for (int32_t row = 0; row < kernel_size; ++row) {
        for (int32_t col = 0; col < kernel_size; ++col) {
            const int32_t dr = row - center;
            const int32_t dc = col - center;
            gaussian_kernel_.kernel_mat(row, col) = std::exp(-0.5f * (dr * dr + dc * dc) / sigma2);
            sum += gaussian_kernel_.kernel_mat(row, col);
        }
    }

    // Normalize kernel to ensure sum of weights = 1
    gaussian_kernel_.kernel_mat /= sum;

    // Precompute second, fourth, and mixed moments of the Gaussian kernel.
    gaussian_kernel_.k2 = 0.0f;
    gaussian_kernel_.k4 = 0.0f;
    gaussian_kernel_.k22 = 0.0f;
    for (int32_t row = 0; row < kernel_size; ++row) {
        for (int32_t col = 0; col < kernel_size; ++col) {
            const int32_t dr = row - center;
            const int32_t dc = col - center;
            const float w = gaussian_kernel_.kernel_mat(row, col);
            gaussian_kernel_.k2 += w * dr * dr;
            gaussian_kernel_.k4 += w * dr * dr * dr * dr;
            gaussian_kernel_.k22 += w * dr * dr * dc * dc;
        }
    }

    return true;
}

bool DenseOpticalFlow::ComputeGaussianWeightedSecondMomentMatrix(const GrayImage &image, const int32_t image_idx) {
    RETURN_FALSE_IF(image.data() == nullptr);
    RETURN_FALSE_IF(image_idx < 0 || image_idx > 1);

    const int rows = image.rows();
    const int cols = image.cols();
    const int half_patch = options_.kHalfPatchSize;

    // Initialize all moment matrices to zero.
    mat_S_0_[image_idx].setZero(rows, cols);
    mat_S_row_[image_idx].setZero(rows, cols);
    mat_S_col_[image_idx].setZero(rows, cols);
    mat_S_rowcol_[image_idx].setZero(rows, cols);
    mat_S_rowrow_[image_idx].setZero(rows, cols);
    mat_S_colcol_[image_idx].setZero(rows, cols);

    // Compute weighted moments over local Gaussian window.
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            for (int dr = -half_patch; dr <= half_patch; ++dr) {
                for (int dc = -half_patch; dc <= half_patch; ++dc) {
                    int r = row + dr;
                    int c = col + dc;

                    // Replicate boundary condition to avoid out-of-bounds access.
                    if (r < 0) {
                        r = 0;
                    } else if (r >= rows) {
                        r = rows - 1;
                    }
                    if (c < 0) {
                        c = 0;
                    } else if (c >= cols) {
                        c = cols - 1;
                    }

                    // Get Gaussian weight and pixel intensity.
                    const float w = gaussian_kernel_.kernel_mat(dr + half_patch, dc + half_patch);
                    const float val = static_cast<float>(image.GetPixelValueNoCheck(r, c));

                    // Accumulate weighted moments.
                    mat_S_0_[image_idx](row, col) += val * w;
                    mat_S_row_[image_idx](row, col) += dr * val * w;
                    mat_S_col_[image_idx](row, col) += dc * val * w;
                    mat_S_rowcol_[image_idx](row, col) += dr * dc * val * w;
                    mat_S_rowrow_[image_idx](row, col) += dr * dr * val * w;
                    mat_S_colcol_[image_idx](row, col) += dc * dc * val * w;
                }
            }
        }
    }

    return true;
}

bool DenseOpticalFlow::ComputeFlowByPixel(const int32_t row, const int32_t col, std::array<Mat, 2> &flow_rc) {
    RETURN_FALSE_IF(row < 0 || row >= mat_S_0_[0].rows() || col < 0 || col >= mat_S_0_[0].cols());

    // Compute polynomial coefficients for reference frame (fixed position)
    Mat2 A1 = Mat2::Zero();
    Vec2 b1 = Vec2::Zero();
    ConstructConstrainFunctionForPixel(row, col, 0, A1, b1);

    // Iterative refinement of flow estimate (Gauss-Newton optimization)
    for (int iter = 0; iter < options_.kMaxIteration; ++iter) {
        // Get current flow vector for the pixel
        const float flow_r = flow_rc[0](row, col);
        const float flow_c = flow_rc[1](row, col);

        // Compute warped coordinates in current frame
        const float sample_r = static_cast<float>(row) + flow_r;
        const float sample_c = static_cast<float>(col) + flow_c;

        // Interpolate coefficients at warped position in current frame
        Mat2 A2 = Mat2::Zero();
        Vec2 b2 = Vec2::Zero();
        ConstructConstrainFunctionForPixel(sample_r, sample_c, 1, A2, b2);

        // Average coefficient matrices and compute difference vector
        const Mat2 A_avg = (A1 + A2) * 0.5f;
        const Vec2 b_diff = b1 - b2;

        // Robust least-squares solution with Tikhonov regularization
        // Solve: (4*A_avg^T*A_avg + λI)Δflow = 2*A_avg^T*b_diff
        const Mat2 M = A_avg * 2.0f;
        const Mat2 MtM = M.transpose() * M;
        const Vec2 Mtb = M.transpose() * b_diff;

        // Adaptive regularization parameter based on matrix trace
        const float lambda = 0.1f * MtM.trace() + 1.0f;
        const Mat2 H = MtM + Mat2::Identity() * lambda;
        const Vec2 delta_flow = H.inverse() * Mtb;

        // Cap update step to prevent numerical instability/explosion
        const float step_norm = delta_flow.norm();
        Vec2 capped_delta = delta_flow;
        if (step_norm > options_.kMaxDeltaFlowStep) {
            capped_delta *= (options_.kMaxDeltaFlowStep / step_norm);
        }

        // Update flow vector with refined estimate.
        flow_rc[0](row, col) += capped_delta(0);
        flow_rc[1](row, col) += capped_delta(1);

        // Stop iteration if convergence threshold reached.
        BREAK_IF(capped_delta.squaredNorm() < options_.kMaxConvergeStep);
    }

    return true;
}

void DenseOpticalFlow::ConstructConstrainFunctionForPixel(int32_t r, int32_t c, int32_t idx, Mat2 &A, Vec2 &b) {
    /* Mathematical Principle (Gunnar Farneback's Algorithm):
     * 1. Local Polynomial Expansion:
     *    The local image intensity f(x) is approximated by a quadratic polynomial:
     *    f(x) ~ x^T * A * x + b^T * x + c
     *    where x = [r, c]^T is the local coordinate relative to the neighborhood center.
     *
     * 2. Weighted Least Squares:
     *    The coefficients (A, b, c) are determined by minimizing:
     *    sum { w(x) * |f(x) - (x^T * A * x + b^T * x + c)|^2 }
     *    where w(x) is the Gaussian weighting kernel.
     *
     * 3. Relationship between Moments and Coefficients:
     *    The precomputed moments (S0, Sr, Sc, Srr, Scc, Src) are Gaussian-weighted sums:
     *    S0 = sum{ w(x) * f(x) }
     *    Sr = sum{ w(x) * r * f(x) }, Sc = sum{ w(x) * c * f(x) }
     *    Srr = sum{ w(x) * r^2 * f(x) }, Scc = sum{ w(x) * c^2 * f(x) }, Src = sum{ w(x) * r * c * f(x) }
     *
     * 4. Solving for A and b:
     *    - Linear term b:
     *      b = [Sr / k2, Sc / k2]^T, where k2 is the second moment of the Gaussian kernel.
     *    - Quadratic term A:
     *      A = [ a,   c/2 ]
     *          [ c/2, b_coeff ]
     *      The coefficients (a, b_coeff) are derived using the fourth moments (k4, k22) of the kernel:
     *      D = k4 - k2^2, E = k22 - k2^2
     *      term1 = (Srr + Scc - 2*k2*S0) / (D + E)
     *      term2 = (Srr - Scc) / (D - E)
     *      a = 0.5 * (term1 + term2), b_coeff = 0.5 * (term1 - term2)
     *      c_coeff = Src / k22
     */
    const float S0 = mat_S_0_[idx](r, c);
    const float Sr = mat_S_row_[idx](r, c);
    const float Sc = mat_S_col_[idx](r, c);
    const float Srr = mat_S_rowrow_[idx](r, c);
    const float Scc = mat_S_colcol_[idx](r, c);
    const float Src = mat_S_rowcol_[idx](r, c);

    const float D = gaussian_kernel_.k4 - gaussian_kernel_.k2 * gaussian_kernel_.k2;
    const float E = gaussian_kernel_.k22 - gaussian_kernel_.k2 * gaussian_kernel_.k2;
    const float inv_D_plus_E = 1.0f / (D + E + 1e-6f);
    const float inv_D_minus_E = 1.0f / (D - E + 1e-6f);

    const float term1 = (Srr + Scc - 2.0f * gaussian_kernel_.k2 * S0) * inv_D_plus_E;
    const float term2 = (Srr - Scc) * inv_D_minus_E;

    const float a = 0.5f * (term1 + term2);
    const float b_coeff = 0.5f * (term1 - term2);
    const float c_coeff = Src / (gaussian_kernel_.k22 + 1e-6f);

    A(0, 0) = a;
    A(0, 1) = 0.5f * c_coeff;
    A(1, 0) = A(0, 1);
    A(1, 1) = b_coeff;
    b(0) = Sr / (gaussian_kernel_.k2 + 1e-6f);
    b(1) = Sc / (gaussian_kernel_.k2 + 1e-6f);
}

void DenseOpticalFlow::ConstructConstrainFunctionForPixel(float r, float c, int32_t idx, Mat2 &A, Vec2 &b) {
    using namespace slam_utility;
    const float S0 = Utility::Interpolate(mat_S_0_[idx], r, c);
    const float Sr = Utility::Interpolate(mat_S_row_[idx], r, c);
    const float Sc = Utility::Interpolate(mat_S_col_[idx], r, c);
    const float Srr = Utility::Interpolate(mat_S_rowrow_[idx], r, c);
    const float Scc = Utility::Interpolate(mat_S_colcol_[idx], r, c);
    const float Src = Utility::Interpolate(mat_S_rowcol_[idx], r, c);

    const float D = gaussian_kernel_.k4 - gaussian_kernel_.k2 * gaussian_kernel_.k2;
    const float E = gaussian_kernel_.k22 - gaussian_kernel_.k2 * gaussian_kernel_.k2;
    const float inv_D_plus_E = 1.0f / (D + E + 1e-6f);
    const float inv_D_minus_E = 1.0f / (D - E + 1e-6f);

    const float term1 = (Srr + Scc - 2.0f * gaussian_kernel_.k2 * S0) * inv_D_plus_E;
    const float term2 = (Srr - Scc) * inv_D_minus_E;

    const float a = 0.5f * (term1 + term2);
    const float b_coeff = 0.5f * (term1 - term2);
    const float c_coeff = Src / (gaussian_kernel_.k22 + 1e-6f);

    A(0, 0) = a;
    A(0, 1) = 0.5f * c_coeff;
    A(1, 0) = A(0, 1);
    A(1, 1) = b_coeff;
    b(0) = Sr / (gaussian_kernel_.k2 + 1e-6f);
    b(1) = Sc / (gaussian_kernel_.k2 + 1e-6f);
}

bool DenseOpticalFlow::SmoothFlow(std::array<Mat, 2> &flow_rc) {
    const int32_t rows = flow_rc[0].rows();
    const int32_t cols = flow_rc[0].cols();
    std::array<Mat, 2> smoothed_flow;
    smoothed_flow[0].resize(rows, cols);
    smoothed_flow[1].resize(rows, cols);

    std::vector<float> window_r, window_c;
    window_r.reserve(9);
    window_c.reserve(9);
    // 3x3 median filter for both row and column flow components.
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
            window_r.clear();
            window_c.clear();
            // Collect 3x3 local neighborhood.
            for (int32_t dr = -1; dr <= 1; ++dr) {
                for (int32_t dc = -1; dc <= 1; ++dc) {
                    // Clamp coordinates to image boundaries.
                    const int32_t nr = std::min(std::max(r + dr, 0), rows - 1);
                    const int32_t nc = std::min(std::max(c + dc, 0), cols - 1);
                    window_r.push_back(flow_rc[0](nr, nc));
                    window_c.push_back(flow_rc[1](nr, nc));
                }
            }
            // Compute median value (5th element in sorted 9-element vector).
            std::nth_element(window_r.begin(), window_r.begin() + 4, window_r.end());
            std::nth_element(window_c.begin(), window_c.begin() + 4, window_c.end());
            smoothed_flow[0](r, c) = window_r[4];
            smoothed_flow[1](r, c) = window_c[4];
        }
    }

    // Replace original flow with smoothed version.
    flow_rc[0] = std::move(smoothed_flow[0]);
    flow_rc[1] = std::move(smoothed_flow[1]);
    return true;
}

}  // namespace feature_tracker
