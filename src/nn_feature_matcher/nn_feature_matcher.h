#ifndef _NN_FEATURE_MATCHER_H_
#define _NN_FEATURE_MATCHER_H_

#include "basic_type.h"
#include "slam_basic_math.h"
#include "slam_operations.h"
#include "onnx_run_time.h"
#include "feature_tracker.h"

namespace FEATURE_TRACKER {

/* Class NNFeatureMatcher Declaration. */
class NNFeatureMatcher {

public:
    enum class ModelType: uint8_t {
        kLightglueForSuperpointScoreMat = 0,
        kLightglueForSuperpointMatches = 1,
    };

    struct Options {
        int32_t kMaxNumberOfMatches = 300;
        float kMinValidMatchScore = -3.0f;
        ModelType kModelType = ModelType::kLightglueForSuperpointScoreMat;
    };

public:
    NNFeatureMatcher() = default;
    virtual ~NNFeatureMatcher() = default;

    bool Initialize();
    template <typename NNFeatureDescriptorType>
    bool Match(const std::vector<NNFeatureDescriptorType> &descriptors_ref,
               const std::vector<NNFeatureDescriptorType> &descriptors_cur,
               const std::vector<Vec2> &pixel_uv_ref,
               const std::vector<Vec2> &pixel_uv_cur,
               std::vector<Vec2> &matched_pixel_uv_cur,
               std::vector<uint8_t> &status);

    // Reference for member variables.
    Options &options() { return options_; }
    // Const reference for member variables.
    const Options &options() const { return options_; }

private:
    template <typename NNFeatureDescriptorType>
    bool InferenceSession(const std::vector<NNFeatureDescriptorType> &descriptors_ref,
                          const std::vector<NNFeatureDescriptorType> &descriptors_cur,
                          const std::vector<Vec2> &pixel_uv_ref,
                          const std::vector<Vec2> &pixel_uv_cur);

private:
    Options options_;

    static Ort::Env onnx_environment_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::RunOptions run_options_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<MatImgF> input_tensor_matrices_;
    std::vector<Ort::Value> input_tensors_;
    std::vector<Ort::Value> output_tensors_;

};

}

#endif // end of _NN_FEATURE_MATCHER_H_