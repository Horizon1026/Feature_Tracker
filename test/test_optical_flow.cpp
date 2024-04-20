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
#include "optical_flow_lssd_klt.h"

using namespace SLAM_VISUALIZOR;

#define DRAW_TRACKING_RESULT (1)

namespace {
    constexpr int32_t kMaxNumberOfFeaturesToTrack = 300;
    constexpr int32_t kHalfPatchSize = 6;
    constexpr FEATURE_TRACKER::OpticalFlowMethod kDefaultMethod = FEATURE_TRACKER::OpticalFlowMethod::kFast;
    constexpr int32_t kMaxPyramidLevel = 4;
}

std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";

void DetectFeatures(const GrayImage &image, std::vector<Vec2> &pixel_uv) {
    FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 25;
    detector.feature().options().kHalfPatchSize = 1;
    detector.feature().options().kMinValidResponse = 40.0f;
    detector.DetectGoodFeatures(image, kMaxNumberOfFeaturesToTrack, pixel_uv);
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
    klt.TrackFeatures(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TockTickInMillisecond();

#if DRAW_TRACKING_RESULT
    // Visualizor::ShowImageWithDetectedFeatures("Basic KLT : Feature before multi tracking", ref_image, ref_pixel_uv);
    Visualizor::ShowImageWithTrackedFeatures("Basic KLT : Feature after multi tracking", cur_image, ref_pixel_uv, cur_pixel_uv, status);
    Visualizor::WaitKey(1);
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
    klt.TrackFeatures(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TockTickInMillisecond();

#if DRAW_TRACKING_RESULT
    // Visualizor::ShowImageWithDetectedFeatures("Affine KLT : Feature before multi tracking", ref_image, ref_pixel_uv);
    Visualizor::ShowImageWithTrackedFeatures("Affine KLT : Feature after multi tracking", cur_image, ref_pixel_uv, cur_pixel_uv, status);
    Visualizor::WaitKey(1);
#endif

    return cost_time;
}

float TestOpticalFlowLssdKlt(int32_t pyramid_level, int32_t patch_size, uint8_t method) {
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

    FEATURE_TRACKER::OpticalFlowLssdKlt klt;
    klt.options().kPatchRowHalfSize = patch_size;
    klt.options().kPatchColHalfSize = patch_size;
    klt.options().kMethod = static_cast<FEATURE_TRACKER::OpticalFlowMethod>(method);
    klt.consider_patch_luminance() = false;

    TickTock timer;
    ref_pyramid.CreateImagePyramid(pyramid_level);
    cur_pyramid.CreateImagePyramid(pyramid_level);
    klt.TrackFeatures(ref_pyramid, cur_pyramid, ref_pixel_uv, cur_pixel_uv, status);
    const float cost_time = timer.TockTickInMillisecond();

#if DRAW_TRACKING_RESULT
    // Visualizor::ShowImageWithDetectedFeatures("Lssd KLT : Feature before multi tracking", ref_image, ref_pixel_uv);
    Visualizor::ShowImageWithTrackedFeatures("Lssd KLT : Feature after multi tracking", cur_image, ref_pixel_uv, cur_pixel_uv, status);
    Visualizor::WaitKey(1);
#endif

    return cost_time;
}

int main(int argc, char **argv) {
    float cost_time = TestOpticalFlowBasicKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Basic klt cost time " << cost_time << " ms.");

    cost_time = TestOpticalFlowAffineKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Affine klt cost time " << cost_time << " ms.");

    cost_time = TestOpticalFlowLssdKlt(kMaxPyramidLevel, kHalfPatchSize, static_cast<uint8_t>(kDefaultMethod));
    ReportInfo("Lssd klt cost time " << cost_time << " ms.");

#if DRAW_TRACKING_RESULT
    Visualizor::WaitKey(0);
#endif // end of DRAW_TRACKING_RESULT
    return 0;
}
