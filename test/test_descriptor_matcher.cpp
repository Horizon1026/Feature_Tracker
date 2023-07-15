#include "iostream"
#include "cstdint"
#include "string"
#include "vector"
#include "ctime"
#include "thread"
#include "random"

#include "feature_point_detector.h"
#include "feature_harris.h"
#include "descriptor_brief.h"
#include "descriptor_matcher.h"

#include "log_report.h"
#include "slam_memory.h"
#include "visualizor.h"

std::string test_ref_image_file_name = "../example/optical_flow/ref_image.png";
std::string test_cur_image_file_name = "../example/optical_flow/cur_image.png";

class BriefMatcher : public FEATURE_TRACKER::DescriptorMatcher<FEATURE_DETECTOR::BriefType> {

public:
    BriefMatcher() : FEATURE_TRACKER::DescriptorMatcher<FEATURE_DETECTOR::BriefType>() {}
    virtual ~BriefMatcher() = default;

    virtual int32_t ComputeDistance(const FEATURE_DETECTOR::BriefType &descriptor_ref,
                                    const FEATURE_DETECTOR::BriefType &descriptor_cur) override {
        if (descriptor_ref.empty() || descriptor_cur.empty()) {
            return kMaxInt32;
        }

        int32_t distance = 0;
        for (uint32_t i = 0; i < descriptor_ref.size(); ++i) {
            if (descriptor_ref[i] != descriptor_cur[i]) {
                ++distance;
            }
        }
        return distance;
    }
};

void TestFeaturePointMatcher() {
    ReportInfo(YELLOW ">> Test Feature Point Matcher." RESET_COLOR);

    // Load images.
    GrayImage ref_image;
    GrayImage cur_image;
    Visualizor::LoadImage(test_ref_image_file_name, ref_image);
    Visualizor::LoadImage(test_cur_image_file_name, cur_image);
    ReportInfo("Load images from " << test_ref_image_file_name << " and " << test_cur_image_file_name);

    // Detect features.
    FEATURE_DETECTOR::FeaturePointDetector<FEATURE_DETECTOR::HarrisFeature> detector;
    detector.options().kMinFeatureDistance = 20;
    detector.feature().options().kMinValidResponse = 40.0f;

    std::vector<Vec2> ref_features, cur_features;
    detector.DetectGoodFeatures(ref_image, 120, ref_features);
    detector.DetectGoodFeatures(cur_image, 120, cur_features);
    ReportInfo("Detect features in two images.");

    // Compute descriptors for these features.
    FEATURE_DETECTOR::BriefDescriptor descriptor;
    descriptor.options().kLength = 256;
    descriptor.options().kHalfPatchSize = 8;

    std::vector<FEATURE_DETECTOR::BriefType> ref_desp, cur_desp;
    descriptor.Compute(ref_image, ref_features, ref_desp);
    descriptor.Compute(cur_image, cur_features, cur_desp);
    ReportInfo("Compute descriptor for all features in ref and cur image.");

    // Match features with descriptors.
    BriefMatcher matcher;
    matcher.options().kMaxValidPredictRowDistance = 50;
    matcher.options().kMaxValidPredictColDistance = 50;
    matcher.options().kMaxValidDescriptorDistance = 60;

    std::vector<Vec2> matched_cur_features;
    std::vector<uint8_t> status;
    const bool res = matcher.NearbyMatch(ref_desp, cur_desp, ref_features, cur_features, matched_cur_features, status);

    int32_t cnt = 0;
    for (uint32_t i = 0; i < status.size(); ++i) {
        cnt += status[i] == static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked);
    }
    ReportInfo("Match features by descriptors, result is " << res << ", tracked features " << cnt << " / " << status.size());

    // Show match result.
    uint8_t *merged_gray_buf = (uint8_t *)SlamMemory::Malloc(cur_image.rows() * cur_image.cols() * 2 * sizeof(uint8_t));
    GrayImage merged_image(merged_gray_buf, cur_image.rows(), cur_image.cols() * 2, true);
    for (int32_t v = 0; v < merged_image.rows(); ++v) {
        for (int32_t u = 0; u < merged_image.cols(); ++u) {
            if (u < cur_image.cols()) {
                merged_image.SetPixelValueNoCheck(v, u, ref_image.GetPixelValueNoCheck(v, u));
            } else {
                merged_image.SetPixelValueNoCheck(v, u, ref_image.GetPixelValueNoCheck(v, u - cur_image.cols()));
            }
        }
    }
    // Construct image to show.
    uint8_t *merged_rgb_buf = (uint8_t *)SlamMemory::Malloc(merged_image.rows() * merged_image.cols() * 3 * sizeof(uint8_t));
    RgbImage show_image(merged_rgb_buf, merged_image.rows(), merged_image.cols(), true);
    Visualizor::ConvertUint8ToRgb(merged_image.data(), show_image.data(), merged_image.rows() * merged_image.cols());

    // [ALL] Draw pairs.
    for (uint32_t i = 0; i < matched_cur_features.size(); ++i) {
        if (status[i] != static_cast<uint8_t>(FEATURE_TRACKER::TrackStatus::kTracked)) {
            continue;
        }
        Visualizor::DrawBressenhanLine(show_image,
            ref_features[i].x(), ref_features[i].y(),
            matched_cur_features[i].x() + cur_image.cols(),
            matched_cur_features[i].y(),
            RgbPixel{.r = static_cast<uint8_t>(std::rand() % 256),
                     .g = static_cast<uint8_t>(std::rand() % 256),
                     .b = static_cast<uint8_t>(std::rand() % 256)});
    }
    // [left] Draw reference points.
    for (uint32_t i = 0; i < ref_features.size(); ++i) {
        Visualizor::DrawSolidCircle(show_image, ref_features[i].x(), ref_features[i].y(),
            3, RgbPixel{.r = 255, .g = 0, .b = 0});
    }
    // [right] Draw result points.
    for (uint32_t i = 0; i < cur_features.size(); ++i) {
        Visualizor::DrawSolidCircle(show_image, cur_features[i].x() + cur_image.cols(), cur_features[i].y(),
            3, RgbPixel{.r = 0, .g = 255, .b = 255});
    }

    Visualizor::ShowImage("Features matched by Brief descriptor", show_image);
    Visualizor::WaitKey(0);
}

int main(int argc, char **argv) {

    TestFeaturePointMatcher();

    return 0;
}
