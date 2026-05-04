#ifndef _DENSE_OPTICAL_FLOW_TRACKER_H
#define _DENSE_OPTICAL_FLOW_TRACKER_H

#include "basic_type.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "slam_basic_math.h"

namespace feature_tracker {

class DenseOpticalFlow {

public:
    struct Options {
        int32_t kMaxIteration = 10;
        int32_t kHalfPatchSize = 3;
    };

public:
    DenseOpticalFlow() = default;
    virtual ~DenseOpticalFlow() = default;

    bool Track(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, std::array<Mat, 2> &flow_rc);
    bool Track(const GrayImage &ref_image, const GrayImage &cur_image, std::array<Mat, 2> &flow_rc);

    std::string OpticalFlowMethodName() const { return "Gunnar Farneback"; }

    // Reference for member variables.
    Options &options() { return options_; }
    Mat &gaussian_kernel() { return gaussian_kernel_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }
    const Mat &gaussian_kernel() const { return gaussian_kernel_; }

private:
    bool TrackMultipleLevel(const ImagePyramid &ref_pyramid, const ImagePyramid &cur_pyramid, std::array<Mat, 2> &flow_rc);
    bool TrackSingleLevel(const GrayImage &ref_image, const GrayImage &cur_image, std::array<Mat, 2> &flow_rc);

    bool InitializeGaussianKernel();
    bool ComputeGaussianWeightedSecondMomentMatrix(const GrayImage &image, const int32_t image_idx);
    bool ComputeFlowByPixel(const int32_t row, const int32_t col, std::array<Mat, 2> &flow_rc);

private:
    Options options_;

    Mat gaussian_kernel_;
    // Gaussian weighted second moment matrix.
    std::array<Mat, 2> mat_S_0_;
    std::array<Mat, 2> mat_S_row_;
    std::array<Mat, 2> mat_S_col_;
    std::array<Mat, 2> mat_S_rowcol_;
    std::array<Mat, 2> mat_S_rowrow_;
    std::array<Mat, 2> mat_S_colcol_;
};

}  // namespace feature_tracker

#endif // end of _DENSE_OPTICAL_FLOW_TRACKER_H_
