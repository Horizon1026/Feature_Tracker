#include "basic_type.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "slam_basic_math.h"
#include "visualizor_2d.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "dense_optical_flow.h"

using namespace slam_visualizor;

std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test dense optical flow." RESET_COLOR);
    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);

    // Generate image pyramids.
    ImagePyramid ref_pyramid, cur_pyramid;
    ref_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * ref_image.rows() * ref_image.cols()), true);
    cur_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * cur_image.rows() * cur_image.cols()), true);
    ref_pyramid.SetRawImage(ref_image.data(), ref_image.rows(), ref_image.cols());
    cur_pyramid.SetRawImage(cur_image.data(), cur_image.rows(), cur_image.cols());

    // Prepare for dense optical flow.
    feature_tracker::DenseOpticalFlow dense_optical_flow;
    dense_optical_flow.options().kHalfPatchSize = 3;
    dense_optical_flow.options().kMaxIteration = 10;

    // Track dense optical flow.
    std::array<Mat, 2> flow_rc;
    dense_optical_flow.Track(ref_image, cur_image, flow_rc);

    ReportInfo("Gaussian kernel is\n" << dense_optical_flow.gaussian_kernel());

    return 0;
}
