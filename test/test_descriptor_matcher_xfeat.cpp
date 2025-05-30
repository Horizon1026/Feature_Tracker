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
using namespace FEATURE_TRACKER;

namespace {
    constexpr int32_t kMaxNumberOfFeaturesToTrack = 200;
    std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
    std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";
}

class XFeatMatcher : public DescriptorMatcher<XFeatDescriptorType> {

public:
    XFeatMatcher() : DescriptorMatcher<XFeatDescriptorType>() {}
    virtual ~XFeatMatcher() = default;

    virtual float ComputeDistance(const XFeatDescriptorType &descriptor_ref,
                                  const XFeatDescriptorType &descriptor_cur) override {
        // Calculate cosine similarity.
        float dot = 0.0f;
        float norm_ref = 0.0f;
        float norm_cur = 0.0f;
        for (uint32_t i = 0; i < descriptor_ref.size(); ++i) {
            dot += descriptor_cur[i] * descriptor_ref[i];
            norm_ref += descriptor_ref[i] * descriptor_ref[i];
            norm_cur += descriptor_cur[i] * descriptor_cur[i];
        }
        norm_ref = std::sqrt(norm_ref);
        norm_cur = std::sqrt(norm_cur);

        const float cosine_similarity = dot / norm_ref / norm_cur;
        return 2.0f - cosine_similarity;
    }

};

void TestFeaturePointMatcher() {
    ReportInfo(YELLOW ">> Test Feature Point Matcher with XFeat." RESET_COLOR);

    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);
    ReportInfo("Load images from " << test_ref_image_file_name << " and " << test_cur_image_file_name);

    // Initialize the detector.
    NNFeaturePointDetector detector("../../Feature_Detector/src/nn_feature_point_detector/models/xfeat_cpu_1_1_h_w.pt");
    detector.options().kModelType = NNFeaturePointDetector::ModelType::kXFeat;
    detector.options().kMinResponse = 0.6f;
    detector.options().kMinFeatureDistance = 10;

    // Detect features and compute descriptors.
    std::vector<Vec2> ref_features, cur_features;
    std::vector<XFeatDescriptorType> ref_descriptors, cur_descriptors;
    detector.DetectGoodFeaturesWithDescriptor(ref_image, kMaxNumberOfFeaturesToTrack, ref_features, ref_descriptors);
    detector.DetectGoodFeaturesWithDescriptor(cur_image, kMaxNumberOfFeaturesToTrack, cur_features, cur_descriptors);
    ReportInfo("Detect " << ref_features.size() << " features in ref image and " << cur_features.size() <<
        " features in cur image.");

    // Match features with descriptors.
    XFeatMatcher matcher;
    matcher.options().kMaxValidPredictRowDistance = 50;
    matcher.options().kMaxValidPredictColDistance = 50;
    matcher.options().kMaxValidDescriptorDistance = 1.2f;

    TickTock timer;
    std::vector<Vec2> matched_cur_features;
    std::vector<uint8_t> status;
    const bool res = matcher.NearbyMatch(ref_descriptors, cur_descriptors, ref_features, cur_features, matched_cur_features, status);
    ReportInfo("Descriptor matcher cost time " << timer.TockTickInMillisecond() << " ms.");

    int32_t cnt = 0;
    for (uint32_t i = 0; i < status.size(); ++i) {
        cnt += status[i] == static_cast<uint8_t>(TrackStatus::kTracked);
    }
    ReportInfo("Match features by descriptors, result is " << res << ", tracked features " << cnt << " / " << status.size());

    // Show match result.
    Visualizor2D::ShowImageWithTrackedFeatures("Features matched by XFeat", ref_image, cur_image,
        ref_features, matched_cur_features, status, static_cast<uint8_t>(TrackStatus::kTracked));
    Visualizor2D::WaitKey(0);
}

int main(int argc, char **argv) {
    TestFeaturePointMatcher();
    return 0;
}
