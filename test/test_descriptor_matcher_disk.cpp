#include "cstdint"
#include "ctime"
#include "iostream"
#include "random"
#include "string"
#include "thread"
#include "vector"

#include "descriptor_matcher.h"
#include "nn_feature_point_detector.h"

#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "visualizor_2d.h"

using namespace SLAM_VISUALIZOR;
using namespace FEATURE_DETECTOR;

namespace {
constexpr int32_t kMaxNumberOfFeaturesToTrack = 300;
std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";
}  // namespace

class DiskMatcher : public FEATURE_TRACKER::DescriptorMatcher<DiskDescriptorType> {

public:
    DiskMatcher()
        : FEATURE_TRACKER::DescriptorMatcher<DiskDescriptorType>() {}
    virtual ~DiskMatcher() = default;

    virtual float ComputeDistance(const DiskDescriptorType &descriptor_ref, const DiskDescriptorType &descriptor_cur) override {
        return 0.5f - descriptor_ref.dot(descriptor_cur) / descriptor_ref.norm() / descriptor_cur.norm() * 0.5f;
    }
};

void TestFeaturePointMatcher() {
    ReportInfo(YELLOW ">> Test Feature Point Matcher with Disk." RESET_COLOR);

    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);
    ReportInfo("Load images from " << test_ref_image_file_name << " and " << test_cur_image_file_name);

    // Initialize feature detector.
    NNFeaturePointDetector detector;
    detector.options().kMinResponse = 0.1f;
    detector.options().kMinFeatureDistance = 20;
    detector.options().kMaxNumberOfDetectedFeatures = kMaxNumberOfFeaturesToTrack;
    detector.options().kModelType = NNFeaturePointDetector::ModelType::kDiskNms;
    detector.options().kMaxImageRows = ref_image.rows();
    detector.options().kMaxImageCols = ref_image.cols();
    detector.Initialize();

    // Detect features.
    std::vector<Vec2> ref_features, cur_features;
    std::vector<DiskDescriptorType> ref_desp, cur_desp;
    detector.DetectGoodFeaturesWithDescriptor(ref_image, ref_features, ref_desp);
    detector.DetectGoodFeaturesWithDescriptor(cur_image, cur_features, cur_desp);
    ReportInfo("Detect features with descritors in two images.");

    // Match features with descriptors.
    DiskMatcher matcher;
    matcher.options().kMaxValidPredictRowDistance = 50;
    matcher.options().kMaxValidPredictColDistance = 50;
    matcher.options().kMaxValidDescriptorDistance = 0.1f;

    std::vector<Vec2> matched_cur_features;
    std::vector<uint8_t> status;
    TickTock timer;
    const bool res = matcher.NearbyMatch(ref_desp, cur_desp, ref_features, cur_features, matched_cur_features, status);
    ReportInfo("Descriptor matcher cost time " << timer.TockTickInMillisecond() << " ms.");

    int32_t cnt = 0;
    for (uint32_t i = 0; i < status.size(); ++i) {
        cnt += status[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked);
    }
    ReportInfo("Match features by descriptors, result is " << res << ", tracked features " << cnt << " / " << status.size());

    // Show match result.
    Visualizor2D::ShowImageWithTrackedFeatures("Features matched by Disk descriptor", ref_image, cur_image, ref_features, matched_cur_features, status,
                                               static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    Visualizor2D::WaitKey(0);
}

int main(int argc, char **argv) {
    TestFeaturePointMatcher();
    return 0;
}
