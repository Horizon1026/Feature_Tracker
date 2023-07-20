#include "iostream"
#include "cstdint"
#include "string"
#include "vector"
#include "ctime"
#include "thread"

#include "log_report.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "visualizor.h"

#include "feature_point_detector.h"
#include "feature_harris.h"

#include "optical_flow_lk.h"
#include "optical_flow_klt.h"

#define DRAW_TRACKING_RESULT (1)
#define DETECT_FEATURES_BY_OPENCV (0)

#include "opencv2/opencv.hpp"

namespace {
    constexpr int32_t kMaxNumberOfFeaturesToTrack = 300;
    constexpr int32_t kHalfPatchSize = 6;
    constexpr FEATURE_TRACKER::OpticalFlowMethod kDefaultMethod = FEATURE_TRACKER::OpticalFlowMethod::kFast;
    constexpr int32_t kMaxPyramidLevel = 4;
}

std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";

void DrawReferenceImage(const GrayImage &image, const std::vector<Vec2> &pixel_uv, const std::string &title) {
    uint8_t *show_ref_image_buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3);
    RgbImage show_ref_image(show_ref_image_buf, image.rows(), image.cols(), true);
    Visualizor::ConvertUint8ToRgb(image.data(), show_ref_image.data(), image.rows() * image.cols());
    for (unsigned long i = 0; i < pixel_uv.size(); i++) {
        Visualizor::DrawSolidCircle(show_ref_image, pixel_uv[i].x(), pixel_uv[i].y(),
            3, RgbPixel{.r = 0, .g = 255, .b = 255});
    }
    Visualizor::ShowImage(title, show_ref_image);
}

void DrawCurrentImage(const GrayImage &image, const std::vector<Vec2> &ref_pixel_uv, const std::vector<Vec2> &cur_pixel_uv, const std::string &title, const std::vector<uint8_t> &status) {
    uint8_t *show_cur_image_buf = (uint8_t *)SlamMemory::Malloc(image.rows() * image.cols() * 3);
    RgbImage show_cur_image(show_cur_image_buf, image.rows(), image.cols(), true);
    Visualizor::ConvertUint8ToRgb(image.data(), show_cur_image.data(), image.rows() * image.cols());
    for (unsigned long i = 0; i < ref_pixel_uv.size(); i++) {
        if (status[i] != static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            Visualizor::DrawSolidCircle(show_cur_image, ref_pixel_uv[i].x(), ref_pixel_uv[i].y(),
                3, RgbPixel{.r = 255, .g = 0, .b = 0});
            continue;
        }
        Visualizor::DrawSolidCircle(show_cur_image, cur_pixel_uv[i].x(), cur_pixel_uv[i].y(),
            3, RgbPixel{.r = 0, .g = 200, .b = 255});
        Visualizor::DrawBressenhanLine(show_cur_image, ref_pixel_uv[i].x(), ref_pixel_uv[i].y(),
            cur_pixel_uv[i].x(), cur_pixel_uv[i].y(),
            RgbPixel{.r = 0, .g = 255, .b = 0});
    }
    Visualizor::ShowImage(title, show_cur_image);
}

void DetectFeatures(const GrayImage &image, std::vector<Vec2> &pixel_uv) {

#if DETECT_FEATURES_BY_OPENCV
	cv::Mat cv_image(image.rows(), image.cols(), CV_8UC1, image.data());
    std::vector<cv::Point2f> ref_corners;
    cv::goodFeaturesToTrack(cv_image, ref_corners, kMaxNumberOfFeaturesToTrack, 0.01, 20);

    pixel_uv.clear();
    for (auto &item : ref_corners) {
        pixel_uv.emplace_back(Vec2(item.x, item.y));
    }

#else
    // Detect features.
    FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 25;
    detector.feature().options().kHalfPatchSize = 6;
    detector.feature().options().kMinValidResponse = 40.0f;
    detector.DetectGoodFeatures(image, kMaxNumberOfFeaturesToTrack, pixel_uv);
#endif
}

float TestLkOpticalFlow(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor::LoadImage(test_cur_image_file_name, cur_image);

    // Generate image pyramids.
    ImagePyramid ref_pyramid, cur_pyramid;
    ref_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * ref_image.rows() * ref_image.cols()), true);
    cur_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * cur_image.rows() * cur_image.cols()), true);
    ref_pyramid.SetRawImage(ref_image.data(), ref_image.rows(), ref_image.cols());
    cur_pyramid.SetRawImage(cur_image.data(), cur_image.rows(), cur_image.cols());

    // Detect features.
    std::vector<Vec2> ref_pixel_uv, cur_pixel_uv;
    std::vector<uint8_t> status;
    DetectFeatures(ref_image, ref_pixel_uv);

    // Use LK optical tracker.
    FEATURE_TRACKER::OpticalFlowLk lk;
    lk.options().kPatchRowHalfSize = patch_size;
    lk.options().kPatchColHalfSize = patch_size;
    lk.options().kMethod = static_cast<FEATURE_TRACKER::OpticalFlowMethod>(method);

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    lk.TrackMultipleLevel(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TickInMillisecond();

#if DRAW_TRACKING_RESULT
    // DrawReferenceImage(ref_image, ref_pixel_uv, "LK : Feature before multi tracking");
    DrawCurrentImage(cur_image, ref_pixel_uv, cur_pixel_uv, "LK : Feature after multi tracking", status);
    Visualizor::WaitKey(0);
#endif

    return cost_time;
}

float TestKltOpticalFlow(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor::LoadImage(test_cur_image_file_name, cur_image);

    // Generate image pyramids.
    ImagePyramid ref_pyramid, cur_pyramid;
    ref_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * ref_image.rows() * ref_image.cols()), true);
    cur_pyramid.SetPyramidBuff((uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * cur_image.rows() * cur_image.cols()), true);
    cur_pyramid.SetRawImage(cur_image.data(), cur_image.rows(), cur_image.cols());
    ref_pyramid.SetRawImage(ref_image.data(), ref_image.rows(), ref_image.cols());

    // Detect features.
    std::vector<Vec2> ref_pixel_uv, cur_pixel_uv;
    std::vector<uint8_t> status;
    DetectFeatures(ref_image, ref_pixel_uv);

    FEATURE_TRACKER::OpticalFlowKlt klt;
    klt.options().kPatchRowHalfSize = patch_size;
    klt.options().kPatchColHalfSize = patch_size;
    klt.options().kMethod = static_cast<FEATURE_TRACKER::OpticalFlowMethod>(method);

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackMultipleLevel(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TickInMillisecond();

#if DRAW_TRACKING_RESULT
    // DrawReferenceImage(ref_image, ref_pixel_uv, "KLT : Feature before multi tracking");
    DrawCurrentImage(cur_image, ref_pixel_uv, cur_pixel_uv, "KLT : Feature after multi tracking", status);
    Visualizor::WaitKey(0);
#endif

    return cost_time;
}

float TestOpencvLkOpticalFlow(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
    cv::Mat cv_ref_image, cv_cur_image;
    cv_ref_image = cv::imread(test_ref_image_file_name, 0);
    cv_cur_image = cv::imread(test_cur_image_file_name, 0);

    GrayImage ref_image(cv_ref_image.data, cv_ref_image.rows, cv_ref_image.cols);

    // Detect features.
    std::vector<cv::Point2f> ref_corners, cur_corners;
    std::vector<Vec2> ref_pixel_uv;
    DetectFeatures(ref_image, ref_pixel_uv);
    for (auto &item : ref_pixel_uv) {
        ref_corners.emplace_back(cv::Point2f(item.x(), item.y()));
    }

    std::vector<uchar> status;
    std::vector<float> errors;

    cv::setNumThreads(1);
    TickTock timer;
    cv::calcOpticalFlowPyrLK(cv_ref_image, cv_cur_image, ref_corners, cur_corners, status, errors,
        cv::Size(2 * patch_size + 1, 2 * patch_size + 1), pyramid_level - 1);
    const float cost_time = timer.TickInMillisecond();

#if DRAW_TRACKING_RESULT
    // cv::Mat show_ref_image(cv_ref_image.rows, cv_ref_image.cols, CV_8UC3);
    // cv::cvtColor(cv_ref_image, show_ref_image, cv::COLOR_GRAY2BGR);
    // for (unsigned long i = 0; i < ref_corners.size(); i++) {
    //     cv::circle(show_ref_image, ref_corners[i], 2, cv::Scalar(255, 255, 0), 3);
    // }
    // cv::imshow("OpenCvLk : Feature before multi tracking", show_ref_image);

    cv::Mat show_cur_image(cv_cur_image.rows, cv_cur_image.cols, CV_8UC3);
    cv::cvtColor(cv_cur_image, show_cur_image, cv::COLOR_GRAY2BGR);
    for (unsigned long i = 0; i < cur_corners.size(); i++) {
        if (status[i] != 1) {
            cv::circle(show_cur_image, cur_corners[i], 2, cv::Scalar(0, 0, 255), 3);
            continue;
        }
        cv::circle(show_cur_image, cur_corners[i], 2, cv::Scalar(0, 255, 255), 3);
        cv::line(show_cur_image, ref_corners[i], cur_corners[i], cv::Scalar(0, 255, 0), 1);
    }
    cv::imshow("OpenCvLk : Feature after multi tracking", show_cur_image);

    cv::waitKey(0);
#endif

    return cost_time;
}

int main(int argc, char **argv) {
    std::thread([&]() {
        const float cost_time = TestOpencvLkOpticalFlow(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
        ReportInfo("cv::calcOpticalFlowPyrLK cost time " << cost_time << " ms.");
    }).join();

    std::thread([&]() {
        const float cost_time = TestLkOpticalFlow(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
        ReportInfo("lk.TrackMultipleLevel cost time " << cost_time << " ms.");
    }).join();

    std::thread([&]() {
        const float cost_time = TestKltOpticalFlow(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
        ReportInfo("klt.TrackMultipleLevel cost time " << cost_time << " ms.");
    }).join();

    return 0;
}
