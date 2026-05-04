#include "dense_optical_flow.h"
#include "slam_operations.h"

namespace feature_tracker {

bool DenseOpticalFlow::Track(const GrayImage &ref_image, const GrayImage &cur_image, std::array<Mat, 2> &flow_rc) {
    RETURN_FALSE_IF(ref_image.data() == nullptr);
    RETURN_FALSE_IF(cur_image.data() == nullptr);
    RETURN_FALSE_IF_FALSE(InitializeGaussianKernel());

    // Compute Gaussian weighted second moment matrix.
    RETURN_FALSE_IF_FALSE(ComputeGaussianWeightedSecondMomentMatrix(ref_image, 0));
    RETURN_FALSE_IF_FALSE(ComputeGaussianWeightedSecondMomentMatrix(cur_image, 1));

    // Compute flow by pixel.
    for (int32_t row = 0; row < ref_image.rows(); ++row) {
        for (int32_t col = 0; col < ref_image.cols(); ++col) {
            ComputeFlowByPixel(row, col, flow_rc);
        }
    }

    return true;
}

bool DenseOpticalFlow::InitializeGaussianKernel() {
    RETURN_FALSE_IF(options_.kHalfPatchSize < 0);

    const int32_t center = options_.kHalfPatchSize;
    const int32_t kernel_size = 2 * center + 1;
    gaussian_kernel_.setZero(kernel_size, kernel_size);
    if (center == 0) {
        gaussian_kernel_(center, center) = 1.0f;
        return true;
    }

    float sum = 0.0f;
    for (int32_t row = 0; row < gaussian_kernel_.rows(); ++row) {
        for (int32_t col = 0; col < gaussian_kernel_.cols(); ++col) {
            const int32_t drow = row - center;
            const int32_t dcol = col - center;
            const float sigma2 = static_cast<float>(center * center);
            gaussian_kernel_(row, col) = std::exp(-0.5f * (drow * drow + dcol * dcol) / sigma2);
            sum += gaussian_kernel_(row, col);
        }
    }

    gaussian_kernel_ /= sum;
    return true;
}

bool DenseOpticalFlow::ComputeGaussianWeightedSecondMomentMatrix(const GrayImage &image, const int32_t image_idx) {
    RETURN_FALSE_IF(image.data() == nullptr);
    RETURN_FALSE_IF(image_idx < 0 || image_idx > 1);

    mat_S_0_[image_idx].setZero(image.rows(), image.cols());
    mat_S_row_[image_idx].setZero(image.rows(), image.cols());
    mat_S_col_[image_idx].setZero(image.rows(), image.cols());
    mat_S_rowcol_[image_idx].setZero(image.rows(), image.cols());
    mat_S_rowrow_[image_idx].setZero(image.rows(), image.cols());
    mat_S_colcol_[image_idx].setZero(image.rows(), image.cols());
    for (int32_t row = 0; row < image.rows(); ++row) {
        for (int32_t col = 0; col < image.cols(); ++col) {
            for (int32_t drow = -options_.kHalfPatchSize; drow <= options_.kHalfPatchSize; ++drow) {
                for (int32_t dcol = -options_.kHalfPatchSize; dcol <= options_.kHalfPatchSize; ++dcol) {
                    int32_t row_i = row + drow;
                    int32_t col_i = col + dcol;
                    // For boundary pixels, use mirror boundary condition.
                    if (row_i < 0) {
                        row_i = - row_i - 1;
                    } else if (row_i >= image.rows()) {
                        row_i = 2 * image.rows() - 1 - row_i;
                    }
                    if (col_i < 0) {
                        col_i = - col_i - 1;
                    } else if (col_i >= image.cols()) {
                        col_i = 2 * image.cols() - 1 - col_i;
                    }
                    // Compute Gaussian weighted second moment matrix.
                    const float gaussian_weight = gaussian_kernel_(drow + options_.kHalfPatchSize, dcol + options_.kHalfPatchSize);
                    mat_S_0_[image_idx](row, col) += image.GetPixelValueNoCheck(row_i, col_i) * gaussian_weight;
                    mat_S_row_[image_idx](row, col) += drow * image.GetPixelValueNoCheck(row_i, col_i) * gaussian_weight;
                    mat_S_col_[image_idx](row, col) += dcol * image.GetPixelValueNoCheck(row_i, col_i) * gaussian_weight;
                    mat_S_rowcol_[image_idx](row, col) += drow * dcol * image.GetPixelValueNoCheck(row_i, col_i) * gaussian_weight;
                    mat_S_rowrow_[image_idx](row, col) += drow * drow * image.GetPixelValueNoCheck(row_i, col_i) * gaussian_weight;
                    mat_S_colcol_[image_idx](row, col) += dcol * dcol * image.GetPixelValueNoCheck(row_i, col_i) * gaussian_weight;
                }
            }
        }
    }

    return true;
}

bool DenseOpticalFlow::ComputeFlowByPixel(const int32_t row, const int32_t col, std::array<Mat, 2> &flow_rc) {
    RETURN_FALSE_IF(row < 0 || row >= mat_S_0_[0].rows() || col < 0 || col >= mat_S_0_[0].cols());

    // Construct local second-order model, including A, b, c.
    std::array<Mat2, 2> A = {Mat2::Zero(), Mat2::Zero()};
    std::array<Vec2, 2> b = {Vec2::Zero(), Vec2::Zero()};
    std::array<float, 2> c = {0.0f, 0.0f};
    for (int32_t image_idx = 0; image_idx < 2; ++image_idx) {
        // Summary of gaussian kernel is 1.0f.
        A[image_idx](0, 0) = mat_S_rowrow_[image_idx](row, col);
        A[image_idx](0, 1) = mat_S_rowcol_[image_idx](row, col);
        A[image_idx](1, 0) = mat_S_rowcol_[image_idx](row, col);
        A[image_idx](1, 1) = mat_S_colcol_[image_idx](row, col);
        b[image_idx](0) = - 2.0f * mat_S_row_[image_idx](row, col);
        b[image_idx](1) = - 2.0f * mat_S_col_[image_idx](row, col);
        c[image_idx] = mat_S_0_[image_idx](row, col);
    }

    // Try to solve the local second-order model.
    const Mat2 average_A = (A[0] + A[1]) / 2.0f;
    const Vec2 diff_b = b[0] - b[1];
    const Mat2 inv_average_A = average_A.llt().solve(Mat2::Identity());
    const Vec2 pixel_flow_rc = - inv_average_A * diff_b * 0.5f;

    flow_rc[0](row, col) = pixel_flow_rc[0];
    flow_rc[1](row, col) = pixel_flow_rc[1];

    return true;
}

}  // namespace feature_tracker
