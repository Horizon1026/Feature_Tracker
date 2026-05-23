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

using namespace slam_visualizor;
using namespace feature_detector;

namespace {
constexpr int32_t kMaxNumberOfFeaturesToTrack = 300;
std::string test_ref_image_file_name = "../example/descriptor_match/ref_image.png";
std::string test_cur_image_file_name = "../example/descriptor_match/cur_image.png";
}  // namespace

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
    std::vector<DiskDescriptorType> ref_desp_disk, cur_desp_disk;
    detector.DetectGoodFeaturesWithDescriptor(ref_image, ref_features, ref_desp_disk);
    detector.DetectGoodFeaturesWithDescriptor(cur_image, cur_features, cur_desp_disk);
    ReportInfo("Detect features with descritors in two images.");

    std::vector<Vec> ref_desp(ref_desp_disk.size());
    std::vector<Vec> cur_desp(cur_desp_disk.size());
    for (uint32_t i = 0; i < ref_desp_disk.size(); ++i) {
        ref_desp[i] = ref_desp_disk[i];
        cur_desp[i] = cur_desp_disk[i];
    }

    // Match features with descriptors.
    feature_tracker::DescriptorMatcher<Vec> matcher;
    matcher.options().kMethod = feature_tracker::DescriptorMatcher<Vec>::MatchMethod::kHungarian;
    matcher.options().kMaxValidPredictRowDistance = 50;
    matcher.options().kMaxValidPredictColDistance = 50;
    // Equivalent to previous distance 0.5 - 0.5 * cosine < 0.1.
    matcher.options().kMinValidDescriptorSimilarity = 0.8f;

    std::vector<Vec2> matched_cur_features;
    std::vector<uint8_t> status;
    TickTock timer;
    std::vector<int32_t> index_pairs_in_cur;
    matcher.Match(ref_desp, cur_desp, ref_features, cur_features, index_pairs_in_cur);
    matcher.FillMatchedPixelByPairIndices(index_pairs_in_cur, cur_features, matched_cur_features, status);
    ReportInfo("Descriptor matcher cost time " << timer.TockTickInMillisecond() << " ms.");

    int32_t cnt = 0;
    for (uint32_t i = 0; i < status.size(); ++i) {
        cnt += status[i] == static_cast<uint8_t>(feature_tracker::TrackStatus::kTracked);
    }
    ReportInfo("Match features by descriptors, tracked features number is " << cnt << " / " << status.size());

    // Show match result.
    Visualizor2D::ShowImageWithTrackedFeatures("Features matched by Disk descriptor", ref_image, cur_image, ref_features, matched_cur_features, status,
                                               static_cast<uint8_t>(feature_tracker::TrackStatus::kTracked));
    Visualizor2D::WaitKey(0);
}

int main(int argc, char **argv) {
    TestFeaturePointMatcher();
    return 0;
}
