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

#include "optical_flow_basic_klt.h"
#include "optical_flow_affine_klt.h"

#define DRAW_TRACKING_RESULT (1)
#define DETECT_FEATURES_BY_OPENCV (0)

#if OPENCV_IS_VALID
#include "opencv2/opencv.hpp"
#endif

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
        if (status[i] != static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked) &&
            status[i] != static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kLargeResidual)) {
            Visualizor::DrawSolidCircle(show_cur_image, cur_pixel_uv[i].x(), cur_pixel_uv[i].y(),
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
#if defined(OPENCV_IS_VALID) && DETECT_FEATURES_BY_OPENCV
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
    detector.feature().options().kHalfPatchSize = 1;
    detector.feature().options().kMinValidResponse = 40.0f;
    detector.DetectGoodFeatures(image, kMaxNumberOfFeaturesToTrack, pixel_uv);
#endif
}

float TestOpticalFlowBasicKlt(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
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
    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    std::vector<uint8_t> status;
    DetectFeatures(ref_image, ref_pixel_uv);
    cur_pixel_uv.reserve(ref_pixel_uv.size());
    status.reserve(ref_pixel_uv.size());

    // Use LK optical tracker.
    FEATURE_TRACKER::OpticalFlowBasicKlt klt;
    klt.options().kPatchRowHalfSize = patch_size;
    klt.options().kPatchColHalfSize = patch_size;
    klt.options().kMethod = static_cast<FEATURE_TRACKER::OpticalFlowMethod>(method);

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackMultipleLevel(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TickInMillisecond();

#if DRAW_TRACKING_RESULT
    // DrawReferenceImage(ref_image, ref_pixel_uv, "Basic KLT : Feature before multi tracking");
    DrawCurrentImage(cur_image, ref_pixel_uv, cur_pixel_uv, "Basic KLT : Feature after multi tracking", status);
    Visualizor::WaitKey(0);
#endif

    return cost_time;
}

float TestOpticalFlowAffineKlt(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
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
    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    std::vector<uint8_t> status;
    DetectFeatures(ref_image, ref_pixel_uv);
    cur_pixel_uv.reserve(ref_pixel_uv.size());
    status.reserve(ref_pixel_uv.size());

    FEATURE_TRACKER::OpticalFlowAffineKlt klt;
    klt.options().kPatchRowHalfSize = patch_size;
    klt.options().kPatchColHalfSize = patch_size;
    klt.options().kMethod = static_cast<FEATURE_TRACKER::OpticalFlowMethod>(method);

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackMultipleLevel(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TickInMillisecond();

#if DRAW_TRACKING_RESULT
    // DrawReferenceImage(ref_image, ref_pixel_uv, "Affine KLT : Feature before multi tracking");
    DrawCurrentImage(cur_image, ref_pixel_uv, cur_pixel_uv, "Affine KLT : Feature after multi tracking", status);
    Visualizor::WaitKey(0);
#endif

    return cost_time;
}

float TestOpencvLkOpticalFlow(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
    float cost_time = 0.0f;

#if OPENCV_IS_VALID
    cv::Mat cv_ref_image, cv_cur_image;
    cv_ref_image = cv::imread(test_ref_image_file_name, 0);
    cv_cur_image = cv::imread(test_cur_image_file_name, 0);

    GrayImage ref_image(cv_ref_image.data, cv_ref_image.rows, cv_ref_image.cols);
    GrayImage cur_image(cv_cur_image.data, cv_cur_image.rows, cv_cur_image.cols);

    // Detect features.
    std::vector<cv::Point2f> ref_corners;
    std::vector<cv::Point2f> cur_corners;
    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    DetectFeatures(ref_image, ref_pixel_uv);
    for (const auto &item : ref_pixel_uv) {
        ref_corners.emplace_back(cv::Point2f(item.x(), item.y()));
    }

    std::vector<uchar> status;
    std::vector<float> errors;
    cur_corners.reserve(ref_pixel_uv.size());
    status.reserve(ref_pixel_uv.size());
    errors.reserve(ref_pixel_uv.size());

    cv::setNumThreads(1);
    TickTock timer;
    cv::calcOpticalFlowPyrLK(cv_ref_image, cv_cur_image, ref_corners, cur_corners, status, errors,
        cv::Size(2 * patch_size + 1, 2 * patch_size + 1), pyramid_level - 1);
    cost_time = timer.TickInMillisecond();

#if DRAW_TRACKING_RESULT
    for (const auto &item : cur_corners) {
        cur_pixel_uv.emplace_back(Vec2(item.x, item.y));
    }
    // DrawReferenceImage(ref_image, ref_pixel_uv, "OpenCV LK : Feature before multi tracking");
    DrawCurrentImage(cur_image, ref_pixel_uv, cur_pixel_uv, "OpenCV LK : Feature after multi tracking", status);
    Visualizor::WaitKey(0);
#endif // end of DRAW_TRACKING_RESULT
#endif // end of OPENCV_IS_VALID

    return cost_time;
}

int main(int argc, char **argv) {
    float cost_time = TestOpencvLkOpticalFlow(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("cv::calcOpticalFlowPyrLK cost time " << cost_time << " ms.");

    cost_time = TestOpticalFlowBasicKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Basic klt cost time " << cost_time << " ms.");

    cost_time = TestOpticalFlowAffineKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Affine klt cost time " << cost_time << " ms.");

    return 0;
}
