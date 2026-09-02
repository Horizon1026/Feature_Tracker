#include "descriptor_matcher.h"

#include "nn_feature_point_detector.h"
#include "slam_log_reporter.h"
#include "slam_memory.h"
#include "tick_tock.h"
#include "visualizor_2d.h"

#include "cstdint"
#include "ctime"
#include "iostream"
#include "random"
#include "string"
#include "thread"
#include "vector"

using namespace slam_visualizor;
using namespace feature_detector;

namespace {
constexpr int32_t kMaxNumberOfFeaturesToTrack = 300;
std::string test_ref_image_file_name = "../example/descriptor_match/ref_image.png";
std::string test_cur_image_file_name = "../example/descriptor_match/cur_image.png";
}  // namespace

void TestFeaturePointMatcher() {
    ReportInfo(YELLOW ">> Test Feature Point Matcher with Superpoint." RESET_COLOR);

    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor2D::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor2D::LoadImage(test_cur_image_file_name, cur_image);
    ReportInfo("Load images from " << test_ref_image_file_name << " and " << test_cur_image_file_name);

    std::vector<Vec2> ref_features, cur_features;
    std::vector<Vec> ref_desp;
    std::vector<Vec> cur_desp;
    NNFeaturePointDetector detector;
    detector.options().kMinResponse = 0.1f;
    detector.options().kMinFeatureDistance = 20;
    detector.options().kMaxNumberOfDetectedFeatures = kMaxNumberOfFeaturesToTrack;
    detector.options().kModelType = NNFeaturePointDetector::ModelType::kSuperpointNms;
    detector.options().kMaxImageRows = ref_image.rows();
    detector.options().kMaxImageCols = ref_image.cols();
    detector.Initialize();

    std::vector<SuperpointDescriptorType> ref_desp_superpoint, cur_desp_superpoint;
    detector.DetectGoodFeaturesWithDescriptor(ref_image, ref_features, ref_desp_superpoint);
    detector.DetectGoodFeaturesWithDescriptor(cur_image, cur_features, cur_desp_superpoint);
    ReportInfo("Detect features with descritors in two images.");

    ref_desp.resize(ref_desp_superpoint.size());
    cur_desp.resize(cur_desp_superpoint.size());
    for (uint32_t i = 0; i < ref_desp_superpoint.size(); ++i) {
        ref_desp[i] = ref_desp_superpoint[i];
    }
    for (uint32_t i = 0; i < cur_desp_superpoint.size(); ++i) {
        cur_desp[i] = cur_desp_superpoint[i];
    }

    // Match features with descriptors.
    TickTock timer;
    feature_tracker::DescriptorMatcher<Vec> matcher;
    matcher.options().matcher.kMethod = feature_tracker::DescriptorMatcher<Vec>::MatchMethod::kNearby;
    matcher.options().matcher.kMinValidDescriptorSimilarity = 0.5f;
    matcher.options().matcher.kMaxDescriptorDistanceRatioBetweenBestAndSecondBest = 0.8f;
    matcher.options().matcher.kEnableCrossCheck = false;
    matcher.options().global_translation.kSearchCol = 150;
    matcher.options().global_translation.kSearchRow = 40;
    matcher.options().displacement_consensus.kEnabled = true;
    matcher.options().displacement_consensus.kTolerance = 25.0f;

    const Vec2 global_shift = matcher.EstimateGlobalTranslation(ref_image, cur_image);
    ReportInfo("Estimated global shift (col, row) = (" << global_shift.x() << ", " << global_shift.y() << ")");

    std::vector<Vec2> pixel_uv_pred_in_cur(ref_features.size());
    for (uint32_t i = 0; i < ref_features.size(); ++i) {
        pixel_uv_pred_in_cur[i] = ref_features[i] + global_shift;
    }

    std::vector<int32_t> index_pairs_in_cur;
    matcher.Match(ref_desp, cur_desp, pixel_uv_pred_in_cur, cur_features, index_pairs_in_cur);

    std::vector<Vec2> matched_cur_features(ref_features.size());
    std::vector<uint8_t> status(ref_features.size(), static_cast<uint8_t>(feature_tracker::TrackStatus::kLargeResidual));
    for (uint32_t i = 0; i < index_pairs_in_cur.size(); ++i) {
        const int32_t j = index_pairs_in_cur[i];
        if (j >= 0 && static_cast<size_t>(j) < cur_features.size()) {
            matched_cur_features[i] = cur_features[static_cast<size_t>(j)];
            status[i] = static_cast<uint8_t>(feature_tracker::TrackStatus::kTracked);
        }
    }
    ReportInfo("Descriptor matcher cost time " << timer.TockTickInMillisecond() << " ms.");

    int32_t cnt = 0;
    for (uint32_t i = 0; i < status.size(); ++i) {
        cnt += status[i] == static_cast<uint8_t>(feature_tracker::TrackStatus::kTracked);
    }
    ReportInfo("Match features by descriptors, tracked features number is " << cnt << " / " << status.size());

    RgbImage match_result_image;
    Visualizor2D::DrawImageWithTrackedFeatures(ref_image, cur_image, ref_features, matched_cur_features, status,
                                               static_cast<uint8_t>(feature_tracker::TrackStatus::kTracked), RgbColor::kCyan, RgbColor::kRed,
                                               match_result_image);
    Visualizor2D::ShowImage("Features matched by Superpoint descriptor", match_result_image);
    Visualizor2D::WaitKey(0);
}

int main(int argc, char **argv) {
    TestFeaturePointMatcher();
    return 0;
}
