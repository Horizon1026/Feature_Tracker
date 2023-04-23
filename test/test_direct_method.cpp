#include "iostream"
#include <cstdint>
#include <string>
#include <vector>
#include <ctime>
#include <thread>

#include "opencv2/opencv.hpp"

#include "log_api.h"
#include "direct_method_tracker.h"

#define FEATURES_TO_TRACK (200)

// Camera intrinsics
float fx = 718.856f, fy = 718.856f, cx = 607.1928f, cy = 185.2157f;
// baseline
float baseline = 0.573f;
std::string test_ref_image_file_name = "../example/direct_method/left.png";
std::string test_ref_depth_file_name = "../example/direct_method/disparity.png";
std::string test_cur_image_file_name = "../example/direct_method/000001.png";
std::array<std::string, 5> test_cur_image_file_names = {
    "../example/direct_method/000001.png",
    "../example/direct_method/000002.png",
    "../example/direct_method/000003.png",
    "../example/direct_method/000004.png",
    "../example/direct_method/000005.png"
};

void TestDirectMethod() {
    // Load images and pyramids.
    cv::Mat cv_ref_image = cv::imread(test_ref_image_file_name, 0);
    cv::Mat cv_ref_depth = cv::imread(test_ref_depth_file_name, 0);

    uint8_t *ref_buf = (uint8_t *)malloc(sizeof(uint8_t) * cv_ref_image.rows * cv_ref_image.cols);
    ImagePyramid ref_pyramid;
    ref_pyramid.SetPyramidBuff(ref_buf);
    ref_pyramid.SetRawImage(cv_ref_image.data, cv_ref_image.rows, cv_ref_image.cols);
    ref_pyramid.CreateImagePyramid(5);

    // Detect features in reference image.
    std::vector<cv::Point2f> cv_ref_corners;
    std::vector<float> ref_pixel_uv_depth;
    cv::goodFeaturesToTrack(cv_ref_image, cv_ref_corners, FEATURES_TO_TRACK, 0.01, 20);
    for (uint32_t i = 0; i < cv_ref_corners.size(); ++i) {
        int32_t disparity = cv_ref_depth.at<uchar>(cv_ref_corners[i].y, cv_ref_corners[i].x);
        ref_pixel_uv_depth.emplace_back(fx * baseline / disparity);
    }

    std::vector<Eigen::Vector2f> ref_pixel_uv, cur_pixel_uv;
    ref_pixel_uv.reserve(cv_ref_corners.size());
    for (uint32_t i = 0; i < cv_ref_corners.size(); ++i) {
        ref_pixel_uv.emplace_back(Eigen::Vector2f(cv_ref_corners[i].x, cv_ref_corners[i].y));
    }

    // Show detected features in reference image.
    cv::Mat show_ref_image(cv_ref_image.rows, cv_ref_image.cols, CV_8UC3);
    cv::cvtColor(cv_ref_image, show_ref_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < cv_ref_corners.size(); i++) {
        cv::circle(show_ref_image, cv::Point2f(ref_pixel_uv[i].x(), ref_pixel_uv[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("Direct method : Feature before multi tracking", show_ref_image);

    // Construct camera intrinsic matrix K.
    std::array<float, 4> K = {fx, fy, cx, cy};

    // Compute features position in reference frame.
    std::vector<Vec3> p_w;
    p_w.reserve(cv_ref_corners.size());
    for (uint32_t i = 0; i < ref_pixel_uv.size(); ++i) {
        p_w.emplace_back(Vec3((ref_pixel_uv[i].x() - cx) / fx, (ref_pixel_uv[i].y() - cy) / fy, 1.0f) * ref_pixel_uv_depth[i]);
    }

    // Construct camera pose in reference frame and current frame;
    Quat q_ref = Quat::Identity();
    Quat q_cur = Quat::Identity();
    Vec3 p_ref = Vec3::Zero();
    Vec3 p_cur = Vec3::Zero();

    for (uint32_t i = 0; i < test_cur_image_file_names.size(); ++i) {
        // Prepare for tracking.
        cv::Mat cv_cur_image = cv::imread(test_cur_image_file_names[i], 0);
        uint8_t *cur_buf = (uint8_t *)malloc(sizeof(uint8_t) * cv_cur_image.rows * cv_cur_image.cols);
        ImagePyramid cur_pyramid;
        cur_pyramid.SetPyramidBuff(cur_buf);
        cur_pyramid.SetRawImage(cv_cur_image.data, cv_cur_image.rows, cv_cur_image.cols);
        cur_pyramid.CreateImagePyramid(5);

        // Construct direct method tracker.
        FEATURE_TRACKER::DirectMethod solver;
        std::vector<uint8_t> status;
        solver.TrackMultipleLevel(ref_pyramid, cur_pyramid, K, q_ref, p_ref, p_w, ref_pixel_uv, cur_pixel_uv, q_cur, p_cur, status);

        // Show result.
        LogInfo("Solved result is q_rc " << LogQuat(q_cur) << ", p_rc " << LogVec(p_cur));

        cv::Mat show_cur_image(cv_cur_image.rows, cv_cur_image.cols, CV_8UC3);
        cv::cvtColor(cv_cur_image, show_cur_image, cv::COLOR_GRAY2BGR);
        for (unsigned long i = 0; i < cv_ref_corners.size(); i++) {
            if (status[i] != static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::TRACKED)) {
                continue;
            }
            cv::circle(show_cur_image, cv::Point2f(cur_pixel_uv[i].x(), cur_pixel_uv[i].y()), 2, cv::Scalar(0, 0, 255), 3);
            cv::line(show_cur_image, cv::Point2f(ref_pixel_uv[i].x(), ref_pixel_uv[i].y()), cv::Point2f(cur_pixel_uv[i].x(), cur_pixel_uv[i].y()), cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("Direct method : Feature after multi tracking", show_cur_image);

        cv::waitKey(0);

        free(cur_buf);
    }
    free(ref_buf);
}

int main(int argc, char **argv) {

    LogInfo(YELLOW ">> Test direct method for all images." RESET_COLOR);
    TestDirectMethod();

    return 0;
}
