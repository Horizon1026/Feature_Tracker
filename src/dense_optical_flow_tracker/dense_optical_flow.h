#ifndef _DENSE_OPTICAL_FLOW_TRACKER_H
#define _DENSE_OPTICAL_FLOW_TRACKER_H

#include "basic_type.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "slam_basic_math.h"

namespace feature_tracker {

/* Class DenseOpticalFlow Declaration. */
class DenseOpticalFlow {

public:
    struct Options {
        int32_t kMaxIteration = 10;
        int32_t kHalfPatchSize = 2;
        float kMaxConvergeStep = 1e-6f;
        float kMaxDeltaFlowStep = 1.0f;
    };

public:
    DenseOpticalFlow() = default;
    virtual ~DenseOpticalFlow() = default;

    bool Track(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, std::array<Mat, 2> &flow_rc);
    bool Track(const GrayImage &ref_image, const GrayImage &cur_image, std::array<Mat, 2> &flow_rc);

    std::string OpticalFlowMethodName() const { return "Gunnar Farneback"; }

    // Reference for parameters.
    Options &options() { return options_; }
    // Const reference for parameters.
    const Options &options() const { return options_; }

private:
    bool InitializeGaussianKernel();
    bool ComputeGaussianWeightedSecondMomentMatrix(const GrayImage &image, const int32_t image_idx);
    bool ComputeFlowByPixel(const int32_t row, const int32_t col, std::array<Mat, 2> &flow_rc);
    bool SmoothFlow(std::array<Mat, 2> &flow_rc);
    void ConstructConstrainFunctionForPixel(int32_t r, int32_t c, int32_t idx, Mat2 &A, Vec2 &b);
    void ConstructConstrainFunctionForPixel(float r, float c, int32_t idx, Mat2 &A, Vec2 &b);

private:
    Options options_;

    struct {
        // Gaussian weighting kernel
        Mat kernel_mat;
        // Precomputed Gaussian kernel statistical moments
        float k2 = 0.0f;
        float k4 = 0.0f;
        float k22 = 0.0f;
    } gaussian_kernel_;

    // Gaussian-weighted second moment matrices (0=ref, 1=current)
    std::array<Mat, 2> mat_S_0_;
    std::array<Mat, 2> mat_S_row_;
    std::array<Mat, 2> mat_S_col_;
    std::array<Mat, 2> mat_S_rowcol_;
    std::array<Mat, 2> mat_S_rowrow_;
    std::array<Mat, 2> mat_S_colcol_;
};

}  // namespace feature_tracker

#endif // end of _DENSE_OPTICAL_FLOW_TRACKER_H_
