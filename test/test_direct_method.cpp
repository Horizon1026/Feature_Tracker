#include "iostream"
#include <cstdint>
#include <string>
#include <vector>
#include <ctime>
#include <thread>

#include "direct_method_tracker.h"

#include "log_report.h"
#include "slam_memory.h"
#include "visualizor.h"
#include "tick_tock.h"

using namespace SLAM_VISUALIZOR;

namespace {
    constexpr int32_t kMaxNumberOfFeaturesToTrack = 300;
}

// Camera intrinsics
const float fx = 718.856f;
const float fy = 718.856f;
const float cx = 607.1928f;
const float cy = 185.2157f;
// baseline
const float baseline = 0.573f;
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
    GrayImage ref_image;
    GrayImage ref_depth;
    Visualizor::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor::LoadImage(test_ref_depth_file_name, ref_depth);

    uint8_t *ref_buf = (uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * ref_image.rows() * ref_image.cols());
    ImagePyramid ref_pyramid;
    ref_pyramid.SetPyramidBuff(ref_buf, true);
    ref_pyramid.SetRawImage(ref_image.data(), ref_image.rows(), ref_image.cols());
    ref_pyramid.CreateImagePyramid(5);

    // Detect features in reference image.
    std::vector<Vec2> ref_pixel_uv;
    std::vector<Vec2> cur_pixel_uv;
    std::vector<float> ref_pixel_uv_depth;
    for (int32_t i = 0; i < kMaxNumberOfFeaturesToTrack; ++i) {
        ref_pixel_uv.emplace_back(Vec2(std::rand() % ref_image.cols(), std::rand() % ref_image.rows()));
        const int32_t disparity = ref_depth.GetPixelValueNoCheck(ref_pixel_uv.back().y(), ref_pixel_uv.back().x());
        ref_pixel_uv_depth.emplace_back(fx * baseline / disparity);
    }

    // Show detected features in reference image.
    Visualizor::ShowImageWithDetectedFeatures("Direct method : Feature before multi tracking", ref_image, ref_pixel_uv);

    // Construct camera intrinsic matrix K.
    std::array<float, 4> K = {fx, fy, cx, cy};

    // Compute features position in reference frame.
    std::vector<Vec3> p_w;
    p_w.reserve(ref_pixel_uv.size());
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
        GrayImage cur_image;
        Visualizor::LoadImage(test_cur_image_file_names[i], cur_image);

        uint8_t *cur_buf = (uint8_t *)SlamMemory::Malloc(sizeof(uint8_t) * cur_image.rows() * cur_image.cols());
        ImagePyramid cur_pyramid;
        cur_pyramid.SetPyramidBuff(cur_buf, true);
        cur_pyramid.SetRawImage(cur_image.data(), cur_image.rows(), cur_image.cols());
        cur_pyramid.CreateImagePyramid(5);

        // Construct direct method tracker.
        TickTock timer;
        FEATURE_TRACKER::DirectMethod solver;
        std::vector<uint8_t> status;
        solver.TrackFeatures(ref_pyramid, cur_pyramid, K, q_ref, p_ref, p_w, ref_pixel_uv, cur_pixel_uv, q_cur, p_cur, status);
        ReportInfo("Direct method cost time " << timer.TockTickInMillisecond() << " ms.");

        // Show result.
        ReportInfo("Solved result is q_rc " << LogQuat(q_cur) << ", p_rc " << LogVec(p_cur));
        Visualizor::ShowImageWithTrackedFeatures("Direct method : Feature after multi tracking", cur_image,
            ref_pixel_uv, cur_pixel_uv, status, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
        Visualizor::WaitKey(0);
    }
}

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test direct method for all images." RESET_COLOR);
    TestDirectMethod();

    return 0;
}
