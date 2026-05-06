#include "basic_type.h"
#include "datatype_image.h"
#include "datatype_image_pyramid.h"
#include "slam_basic_math.h"
#include "visualizor_2d.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "dense_optical_flow.h"
#include "image_painter.h"

#include "enable_stack_backward.h"

using namespace slam_visualizor;

std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test dense optical flow." RESET_COLOR);
    // Load reference and current gray images from disk
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);

    // Create and initialize image pyramids for coarse-to-fine flow estimation
    ImagePyramid ref_pyramid, cur_pyramid;
    ref_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * ref_image.rows() * ref_image.cols()), true);
    cur_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * cur_image.rows() * cur_image.cols()), true);
    ref_pyramid.SetRawImage(ref_image.data(), ref_image.rows(), ref_image.cols());
    cur_pyramid.SetRawImage(cur_image.data(), cur_image.rows(), cur_image.cols());
    // Generate 5-level image pyramid
    ref_pyramid.CreateImagePyramid(5);
    cur_pyramid.CreateImagePyramid(5);

    // Initialize Farneback dense optical flow processor
    feature_tracker::DenseOpticalFlow dense_optical_flow;
    dense_optical_flow.options().kHalfPatchSize = 2;       // 5x5 local patch
    dense_optical_flow.options().kMaxIteration = 20;      // More iterations for accuracy

    // Compute dense optical flow between image pyramids
    std::array<Mat, 2> flow_rc;
    dense_optical_flow.Track(ref_pyramid, cur_pyramid, flow_rc);

    // Sample grid points to visualize flow vectors
    std::vector<Vec2> ref_pts;
    std::vector<Vec2> cur_pts;
    std::vector<uint8_t> status;

    // Grid sampling step size (controls visualization density)
    const int32_t step = 15;
    for (int32_t r = step; r < ref_image.rows() - step; r += step) {
        for (int32_t c = step; c < ref_image.cols() - step; c += step) {
            const float dr = flow_rc[0](r, c);
            const float dc = flow_rc[1](r, c);

            // Store reference point (u,v order: column, row) and tracked point
            ref_pts.emplace_back(c, r);
            cur_pts.emplace_back(c + dc, r + dr);
            status.push_back(1);
        }
    }

    // Show image with feature tracks: start points, end points, and flow lines.
    Visualizor2D::ShowImageWithTrackedFeatures("Dense Flow Vectors", cur_image, ref_pts, cur_pts, status, 1, RgbColor::kCyan, RgbColor::kRed, RgbColor::kGreen);
    Visualizor2D::WaitKey(0);
    return 0;
}
