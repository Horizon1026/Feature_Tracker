#include "iostream"
#include <cstdint>
#include <string>
#include <vector>
#include <ctime>
#include <thread>

#include "opencv2/opencv.hpp"
#include "gperftools/profiler.h"

#include "datatype_basic.h"
#include "optical_flow_datatype.h"
#include "optical_flow_lk.h"

std::string test_ref_image_file_name = "../example/ref_image.png";
std::string test_cur_image_file_name = "../example/cur_image.png";

void test_lk_multi(int32_t pyramid_level, int32_t patch_size, uint8_t method, uint32_t times) {
    cv::Mat cv_ref_image, cv_cur_image;
    cv_ref_image = cv::imread(test_ref_image_file_name, 0);
    cv_cur_image = cv::imread(test_cur_image_file_name, 0);

    ImagePyramid ref_pyramid, cur_pyramid;
    ref_pyramid.SetPyramidBuff((uint8_t *)malloc(sizeof(uint8_t) * cv_ref_image.rows * cv_ref_image.cols * 2));
    cur_pyramid.SetPyramidBuff((uint8_t *)malloc(sizeof(uint8_t) * cv_cur_image.rows * cv_cur_image.cols * 2));
    cur_pyramid.SetRawImage(cv_cur_image.data, cv_cur_image.rows, cv_cur_image.cols);
    ref_pyramid.SetRawImage(cv_ref_image.data, cv_ref_image.rows, cv_ref_image.cols);

    std::vector<cv::Point2f> ref_corners;
    cv::goodFeaturesToTrack(cv_ref_image, ref_corners, 200, 0.01, 20);

    OPTICAL_FLOW::OpticalFlowLk lk;
    std::vector<Eigen::Vector2f> ref_points, cur_points;
    std::vector<OPTICAL_FLOW::TrackStatus> status;
    ref_points.reserve(ref_corners.size());
    for (uint32_t i = 0; i < ref_corners.size(); ++i) {
        ref_points.emplace_back(Eigen::Vector2f(ref_corners[i].x, ref_corners[i].y));
    }

    lk.options().kPatchRowHalfSize = patch_size;
    lk.options().kPatchColHalfSize = patch_size;
    lk.options().kMethod = static_cast<OPTICAL_FLOW::LkMethod>(method);

    ProfilerStart("test.prof");
    for (SLAM_UTILITY::uint32_t i = 0; i < times; ++i) {
        ref_pyramid.CreateImagePyramid(pyramid_level);
        cur_pyramid.CreateImagePyramid(pyramid_level);
        lk.TrackMultipleLevel(&ref_pyramid, &cur_pyramid, ref_points, cur_points, status);
    }
    ProfilerStop();

    free(ref_pyramid.pyramid_buf());
    free(cur_pyramid.pyramid_buf());

}

int main() {
    uint32_t test_times = 1000;
    const uint8_t optical_flow_method = 0;
    const int32_t pyramid_level = 4;
    const int32_t half_patch_size = 6;

    test_lk_multi(pyramid_level, half_patch_size, optical_flow_method, test_times);

    return 0;
}