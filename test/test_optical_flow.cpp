#include "optical_flow_affine_klt.h"
#include "optical_flow_basic_klt.h"
#include "optical_flow_lssd_klt.h"

#include "image_painter.h"
#include "feature_point_detector.h"
#include "feature_point_harris_detector.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "visualizor_2d.h"

#include "cmath"
#include "cstdint"
#include "ctime"
#include "iostream"
#include "string"
#include "thread"
#include "vector"

using namespace slam_visualizor;

#define DRAW_TRACKING_RESULT (1)

namespace {
constexpr int32_t kMaxNumberOfFeaturesToTrack = 300;
constexpr int32_t kHalfPatchSize = 6;
constexpr feature_tracker::OpticalFlow::Method kDefaultMethod = feature_tracker::OpticalFlow::Method::kFast;
constexpr int32_t kMaxPyramidLevel = 6;

#if DRAW_TRACKING_RESULT
void ShowTrackingResultWithGroundTruth(const std::string &window_title, const GrayImage &cur_image,
                                       const std::vector<Vec2> &ref_pixel_uv, const std::vector<Vec2> &cur_pixel_uv,
                                       const std::vector<uint8_t> &status) {
    RgbImage show_image;
    Visualizor2D::DrawImageWithTrackedFeatures(cur_image, ref_pixel_uv, cur_pixel_uv, status,
                                              static_cast<uint8_t>(feature_tracker::TrackStatus::kTracked),
                                              RgbColor::kCyan, RgbColor::kRed, RgbColor::kGreen, show_image);

    // Ground truth for synthetic_ref.png -> synthetic_cur.png: rotate about (width/2, height/2), then translate.
    constexpr float angle = 5.0f * kDegToRad;
    Mat2 rotation = Mat2::Zero();
    rotation << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    const Vec2 center(cur_image.cols() * 0.5f, cur_image.rows() * 0.5f);
    const Vec2 translation(30.0f, 8.0f);
    for (const Vec2 &ref_pixel : ref_pixel_uv) {
        const Vec2 ground_truth = rotation * (ref_pixel - center) + center + translation;
        if (ground_truth.x() < 0 || ground_truth.x() > cur_image.cols() - 1 ||
            ground_truth.y() < 0 || ground_truth.y() > cur_image.rows() - 1) {
            continue;
        }
        image_painter::ImagePainter::DrawHollowCircle(show_image, static_cast<int32_t>(std::lround(ground_truth.x())),
                                                     static_cast<int32_t>(std::lround(ground_truth.y())), 7, RgbColor::kYellow);
    }
    Visualizor2D::ShowImage(window_title + " (yellow circles: ground truth)", show_image);
}
#endif
}  // namespace

std::string test_ref_image_file_name = "../example/optical_flow/synthetic_ref.png";
std::string test_cur_image_file_name = "../example/optical_flow/synthetic_cur.png";

void DetectFeatures(const GrayImage &image, std::vector<Vec2> &pixel_uv) {
    feature_detector::FeaturePointHarrisDetector detector;
    detector.options().kMinFeatureDistance = 25;
    detector.options().kMinValidResponse = 40.0f;
    detector.DetectGoodFeatures(image, kMaxNumberOfFeaturesToTrack, pixel_uv);
}

float TestOpticalFlowBasicKlt(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
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

    // Detect features.
    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    std::vector<uint8_t> status;
    DetectFeatures(ref_image, ref_pixel_uv);
    cur_pixel_uv.reserve(ref_pixel_uv.size());
    status.reserve(ref_pixel_uv.size());

    // Use LK optical tracker.
    feature_tracker::OpticalFlowBasicKlt klt;
    klt.options().kPatchRowHalfSize = patch_size;
    klt.options().kPatchColHalfSize = patch_size;
    klt.options().kMethod = static_cast<feature_tracker::OpticalFlow::Method>(method);

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackFeatures(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TockTickInMillisecond();

#if DRAW_TRACKING_RESULT
    // Visualizor2D::ShowImageWithDetectedFeatures("Basic KLT : Feature before multi tracking", ref_image,
    // ref_pixel_uv);
    ShowTrackingResultWithGroundTruth("Basic KLT : Feature after multi tracking", cur_image, ref_pixel_uv, cur_pixel_uv, status);
    Visualizor2D::WaitKey(1);
#endif

    return cost_time;
}

float TestOpticalFlowAffineKlt(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);

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

    feature_tracker::OpticalFlowAffineKlt klt;
    klt.options().kPatchRowHalfSize = patch_size;
    klt.options().kPatchColHalfSize = patch_size;
    klt.options().kMethod = static_cast<feature_tracker::OpticalFlow::Method>(method);

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackFeatures(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TockTickInMillisecond();

#if DRAW_TRACKING_RESULT
    // Visualizor2D::ShowImageWithDetectedFeatures("Affine KLT : Feature before multi tracking", ref_image,
    // ref_pixel_uv);
    ShowTrackingResultWithGroundTruth("Affine KLT : Feature after multi tracking", cur_image, ref_pixel_uv, cur_pixel_uv, status);
    Visualizor2D::WaitKey(1);
#endif

    return cost_time;
}

float TestOpticalFlowLssdKlt(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);

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

    feature_tracker::OpticalFlowLssdKlt klt;
    klt.options().kPatchRowHalfSize = patch_size;
    klt.options().kPatchColHalfSize = patch_size;
    klt.options().kMethod = static_cast<feature_tracker::OpticalFlow::Method>(method);

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackFeatures(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const double cost_time = timer.TockTickInMillisecond();

#if DRAW_TRACKING_RESULT
    // Visualizor2D::ShowImageWithDetectedFeatures("Lssd KLT : Feature before multi tracking", ref_image, ref_pixel_uv);
    ShowTrackingResultWithGroundTruth("Lssd KLT : Feature after multi tracking", cur_image, ref_pixel_uv, cur_pixel_uv, status);
    Visualizor2D::WaitKey(1);
#endif

    return cost_time;
}

int main(int argc, char **argv) {
    double cost_time = TestOpticalFlowBasicKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Basic klt cost time " << cost_time << " ms.");

    cost_time = TestOpticalFlowAffineKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Affine klt cost time " << cost_time << " ms.");

    cost_time = TestOpticalFlowLssdKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Lssd klt cost time " << cost_time << " ms.");

#if DRAW_TRACKING_RESULT
    Visualizor2D::WaitKey(0);
#endif  // end of DRAW_TRACKING_RESULT
    return 0;
}
