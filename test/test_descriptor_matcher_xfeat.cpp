#include "iostream"
#include "cstdint"
#include "string"
#include "vector"
#include "ctime"
#include "thread"
#include "random"

#include "nn_feature_point_detector.h"
#include "descriptor_matcher.h"

#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "visualizor_2d.h"
#include "tick_tock.h"

using namespace SLAM_VISUALIZOR;
using namespace FEATURE_DETECTOR;

namespace {
    constexpr int32_t kMaxNumberOfFeaturesToTrack = 100;
    std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
    std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";
}

class XFeatMatcher : public FEATURE_TRACKER::DescriptorMatcher<FEATURE_DETECTOR::XFeatType> {

public:
    XFeatMatcher() : FEATURE_TRACKER::DescriptorMatcher<FEATURE_DETECTOR::XFeatType>() {}
    virtual ~XFeatMatcher() = default;

    virtual int32_t ComputeDistance(const FEATURE_DETECTOR::XFeatType &descriptor_ref,
                                    const FEATURE_DETECTOR::XFeatType &descriptor_cur) override {
        float sum = 0.0f;
        for (uint32_t i = 0; i < descriptor_ref.size(); ++i) {
            sum += std::abs(descriptor_ref[i] - descriptor_cur[i]);
        }
        return sum;
    }

};

void TestFeaturePointMatcher() {
    ReportInfo(YELLOW ">> Test Feature Point Matcher." RESET_COLOR);

    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);
    ReportInfo("Load images from " << test_ref_image_file_name << " and " << test_cur_image_file_name);

    // Initialize the detector.
    NNFeaturePointDetector detector("../../Feature_Detector/src/nn_feature_point_detector/models/xfeat_cpu_1_1_h_w.pt");

    // Detect features and compute descriptors.
    std::vector<Vec2> ref_features, cur_features;
    std::vector<FEATURE_DETECTOR::XFeatType> ref_descriptors, cur_descriptors;
    detector.DetectGoodFeatures(ref_image, kMaxNumberOfFeaturesToTrack, ref_features);
    detector.ExtractDescriptors<64>(ref_features, ref_descriptors);
    detector.DetectGoodFeatures(cur_image, kMaxNumberOfFeaturesToTrack, cur_features);
    detector.ExtractDescriptors<64>(cur_features, cur_descriptors);

    // Match features with descriptors.
    XFeatMatcher matcher;
    matcher.options().kMaxValidPredictRowDistance = 50;
    matcher.options().kMaxValidPredictColDistance = 50;
    matcher.options().kMaxValidDescriptorDistance = 60;

    TickTock timer;
    std::vector<Vec2> matched_cur_features;
    std::vector<uint8_t> status;
    const bool res = matcher.NearbyMatch(ref_descriptors, cur_descriptors, ref_features, cur_features, matched_cur_features, status);
    ReportInfo("Descriptor matcher cost time " << timer.TockTickInMillisecond() << " ms.");

    int32_t cnt = 0;
    for (uint32_t i = 0; i < status.size(); ++i) {
        cnt += status[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked);
    }
    ReportInfo("Match features by descriptors, result is " << res << ", tracked features " << cnt << " / " << status.size());

    // Show match result.
    Visualizor2D::ShowImageWithTrackedFeatures("Features matched by XFeat", ref_image, cur_image,
        ref_features, matched_cur_features, status, static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked));
    Visualizor2D::WaitKey(0);
}

int main(int argc, char **argv) {
    TestFeaturePointMatcher();
    return 0;
}
