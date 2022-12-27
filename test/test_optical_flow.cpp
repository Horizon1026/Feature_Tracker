#include "iostream"
#include <cstdint>
#include <string>
#include <vector>
#include <ctime>

#include "opencv2/opencv.hpp"

#include "optical_flow_datatype.h"
#include "optical_flow_lk.h"
#include "optical_flow_klt.h"

#define CONFIG_OPENCV_DRAW (1)

std::string test_ref_image_file_name = "../example/ref_image.png";
std::string test_cur_image_file_name = "../example/cur_image.png";

void test_image() {
    std::cout << ">> test_image" << std::endl;

    cv::Mat cv_image;
    cv_image = cv::imread(test_ref_image_file_name, 0);

    Image image;
    image.SetImage(cv_image.data, cv_image.rows, cv_image.cols);
    std::cout << image.rows() << std::endl;
    std::cout << image.cols() << std::endl;

    float value;
    uint16_t int_value;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            image.GetPixelValue(i, j, &int_value);
            image.GetPixelValue(i, j, &value);
            std::cout << "test image.GetPixelValue is " << int_value << ", " << value << std::endl;
        }
    }

#if CONFIG_OPENCV_DRAW
    cv::Mat show_image(image.rows(), image.cols(), CV_8UC1, image.image_data());
    cv::imshow("convert cv_image to image", show_image);
    cv::waitKey(0);
#endif
}

void test_pyramid() {
    std::cout << ">> test_pyramid" << std::endl;

    cv::Mat cv_image;
    cv_image = cv::imread(test_ref_image_file_name, 0);

    ImagePyramid pyramid;
    pyramid.SetPyramidBuff((uint8_t *)malloc(sizeof(uint8_t) * cv_image.rows * cv_image.cols * 2));
    pyramid.SetRawImage(cv_image.data, cv_image.rows, cv_image.cols);
    pyramid.CreateImagePyramid(5);

#if CONFIG_OPENCV_DRAW
    for (uint32_t i = 0; i < pyramid.level(); ++i) {
        Image one_level = pyramid.GetImage(i);
        cv::Mat image(one_level.rows(), one_level.cols(), CV_8UC1, one_level.image_data());
        cv::imshow(std::to_string(i), image);
        cv::waitKey(1);
    }
    cv::waitKey(0);
#endif
}

void test_lk_multi(int32_t pyramid_level = 4, int32_t patch_size = 4) {
    std::cout << ">> test_lk_multi" << std::endl;
    std::cout << "  pyramid_level is " << pyramid_level << ", patch_size is " << patch_size << std::endl;

    cv::Mat cv_ref_image, cv_cur_image;
    cv_ref_image = cv::imread(test_ref_image_file_name, 0);
    cv_cur_image = cv::imread(test_cur_image_file_name, 0);

    ImagePyramid ref_pyramid, cur_pyramid;
    ref_pyramid.SetPyramidBuff((uint8_t *)malloc(sizeof(uint8_t) * cv_ref_image.rows * cv_ref_image.cols * 2));
    cur_pyramid.SetPyramidBuff((uint8_t *)malloc(sizeof(uint8_t) * cv_cur_image.rows * cv_cur_image.cols * 2));
    cur_pyramid.SetRawImage(cv_cur_image.data, cv_cur_image.rows, cv_cur_image.cols);
    ref_pyramid.SetRawImage(cv_ref_image.data, cv_ref_image.rows, cv_ref_image.cols);

    std::vector<cv::Point2f> new_corners;
    cv::goodFeaturesToTrack(cv_ref_image, new_corners, 200, 0.01, 20);

    OPTICAL_FLOW::OpticalFlowLk lk;
    std::vector<Eigen::Vector2f> ref_points, cur_points;
    std::vector<OPTICAL_FLOW::TrackStatus> status;
    ref_points.reserve(new_corners.size());
    for (uint32_t i = 0; i < new_corners.size(); ++i) {
        ref_points.emplace_back(Eigen::Vector2f(new_corners[i].x, new_corners[i].y));
    }

    lk.options().kPatchRowHalfSize = pyramid_level;
    lk.options().kPatchColHalfSize = pyramid_level;
    lk.options().kMethod = OPTICAL_FLOW::LK_INVERSE_LSE;

    std::chrono::time_point<std::chrono::system_clock> begin, end;
    begin = std::chrono::system_clock::now();
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    lk.TrackMultipleLevel(&ref_pyramid, &cur_pyramid, ref_points, cur_points, status);
    end = std::chrono::system_clock::now();
    std::cout << "lk.TrackMultipleLevel cost time " << std::chrono::duration<double>(end - begin).count() * 1000 << " ms." << std::endl;

#if CONFIG_OPENCV_DRAW
    cv::Mat show_ref_image(cv_ref_image.rows, cv_ref_image.cols, CV_8UC3);
    cv::cvtColor(cv_ref_image, show_ref_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < new_corners.size(); i++) {
        cv::circle(show_ref_image, cv::Point2f(ref_points[i].x(), ref_points[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("LK : Feature before multi tracking", show_ref_image);

    cv::Mat show_cur_image(cv_cur_image.rows, cv_cur_image.cols, CV_8UC3);
    cv::cvtColor(cv_cur_image, show_cur_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < new_corners.size(); i++) {
        if (status[i] != OPTICAL_FLOW::TRACKED) {
            continue;
        }
        cv::circle(show_cur_image, cv::Point2f(cur_points[i].x(), cur_points[i].y()), 2, cv::Scalar(0, 0, 255), 3);
        cv::line(show_cur_image, cv::Point2f(ref_points[i].x(), ref_points[i].y()), cv::Point2f(cur_points[i].x(), cur_points[i].y()), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("LK : Feature after multi tracking", show_cur_image);

    cv::waitKey(0);
#endif

    free(ref_pyramid.pyramid_buf());
    free(cur_pyramid.pyramid_buf());
}



void test_klt_multi(int32_t pyramid_level = 4, int32_t patch_size = 4) {
    std::cout << ">> test_klt_multi" << std::endl;
    std::cout << "  pyramid_level is " << pyramid_level << ", patch_size is " << patch_size << std::endl;

    cv::Mat cv_ref_image, cv_cur_image;
    cv_ref_image = cv::imread(test_ref_image_file_name, 0);
    cv_cur_image = cv::imread(test_cur_image_file_name, 0);

    ImagePyramid ref_pyramid, cur_pyramid;
    ref_pyramid.SetPyramidBuff((uint8_t *)malloc(sizeof(uint8_t) * cv_ref_image.rows * cv_ref_image.cols * 2));
    cur_pyramid.SetPyramidBuff((uint8_t *)malloc(sizeof(uint8_t) * cv_cur_image.rows * cv_cur_image.cols * 2));
    cur_pyramid.SetRawImage(cv_cur_image.data, cv_cur_image.rows, cv_cur_image.cols);
    ref_pyramid.SetRawImage(cv_ref_image.data, cv_ref_image.rows, cv_ref_image.cols);

    std::vector<cv::Point2f> new_corners;
    cv::goodFeaturesToTrack(cv_ref_image, new_corners, 200, 0.01, 20);

    OPTICAL_FLOW::OpticalFlowKlt klt;
    std::vector<Eigen::Vector2f> ref_points, cur_points;
    std::vector<OPTICAL_FLOW::TrackStatus> status;
    ref_points.reserve(new_corners.size());
    for (uint32_t i = 0; i < new_corners.size(); ++i) {
        ref_points.emplace_back(Eigen::Vector2f(new_corners[i].x, new_corners[i].y));
    }

    klt.options().kPatchRowHalfSize = pyramid_level;
    klt.options().kPatchColHalfSize = pyramid_level;
    klt.options().kMethod = OPTICAL_FLOW::KLT_DIRECT;

    std::chrono::time_point<std::chrono::system_clock> begin, end;
    begin = std::chrono::system_clock::now();
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackMultipleLevel(&ref_pyramid, &cur_pyramid, ref_points, cur_points, status);
    end = std::chrono::system_clock::now();
    std::cout << "klt.TrackMultipleLevel cost time " << std::chrono::duration<double>(end - begin).count() * 1000 << " ms." << std::endl;

#if CONFIG_OPENCV_DRAW
    cv::Mat show_ref_image(cv_ref_image.rows, cv_ref_image.cols, CV_8UC3);
    cv::cvtColor(cv_ref_image, show_ref_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < new_corners.size(); i++) {
        cv::circle(show_ref_image, cv::Point2f(ref_points[i].x(), ref_points[i].y()), 2, cv::Scalar(255, 255, 0), 3);
    }
    cv::imshow("KLT : Feature before multi tracking", show_ref_image);

    cv::Mat show_cur_image(cv_cur_image.rows, cv_cur_image.cols, CV_8UC3);
    cv::cvtColor(cv_cur_image, show_cur_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < new_corners.size(); i++) {
        if (status[i] != OPTICAL_FLOW::TRACKED) {
            continue;
        }
        cv::circle(show_cur_image, cv::Point2f(cur_points[i].x(), cur_points[i].y()), 2, cv::Scalar(0, 0, 255), 3);
        cv::line(show_cur_image, cv::Point2f(ref_points[i].x(), ref_points[i].y()), cv::Point2f(cur_points[i].x(), cur_points[i].y()), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("KLT : Feature after multi tracking", show_cur_image);

    cv::waitKey(0);
#endif

    free(ref_pyramid.pyramid_buf());
    free(cur_pyramid.pyramid_buf());
}

int main() {
    uint32_t test_times = 10;

#if CONFIG_OPENCV_DRAW
    test_times = 1;
#endif

    // test_image();
    // test_pyramid();

    for (uint32_t i = 0; i < test_times; ++i) {
        test_lk_multi(4, 6);
    }

    for (uint32_t i = 0; i < test_times; ++i) {
        test_klt_multi(4, 6);
    }

    return 0;
}