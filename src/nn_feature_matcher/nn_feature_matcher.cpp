#include "nn_feature_matcher.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "tick_tock.h"

namespace FEATURE_TRACKER {

Ort::Env NNFeatureMatcher::onnx_environment_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "NNFeatureMatcher");

bool NNFeatureMatcher::Initialize() {
    const std::string model_root_path = "../../Feature_Tracker/onnx_models/";
    std::string model_path;
    switch (options_.kModelType) {
        default:
        case ModelType::kLightglueForSuperpoint: {
            model_path = model_root_path + "superpoint_lightglue.onnx";
            break;
        }
        case ModelType::kLightglueForSuperpointFused: {
            model_path = model_root_path + "superpoint_lightglue_fused.onnx";
            break;
        }
    }

    // Initialize session options if needed.
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    // For onnxruntime of version 1.20, enable GPU.
    OnnxRuntime::TryToEnableCuda(session_options_);

    // Create session.
    try {
        session_ = Ort::Session(NNFeatureMatcher::onnx_environment_, model_path.c_str(), session_options_);
        OnnxRuntime::ReportInformationOfSession(session_);
        ReportInfo("[NNFeatureMatcher] Succeed to load onnx model: " << model_path);
    } catch (const Ort::Exception &e) {
        ReportError("[NNFeatureMatcher] Failed to load onnx model: " << model_path);
    }
    OnnxRuntime::GetSessionIO(session_, input_names_, output_names_);
    input_tensor_matrices_.resize(input_names_.size());
    input_tensors_.clear();
    for (uint32_t i = 0; i < input_names_.size(); ++i) {
        input_tensors_.emplace_back(Ort::Value(nullptr));
    }
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    return true;
}

template bool NNFeatureMatcher::InferenceSession<SuperpointDescriptorType>(const std::vector<SuperpointDescriptorType> &descriptors_ref,
    const std::vector<SuperpointDescriptorType> &descriptors_cur, const std::vector<Vec2> &pixel_uv_ref, const std::vector<Vec2> &pixel_uv_cur);
template bool NNFeatureMatcher::InferenceSession<DiskDescriptorType>(const std::vector<DiskDescriptorType> &descriptors_ref,
    const std::vector<DiskDescriptorType> &descriptors_cur, const std::vector<Vec2> &pixel_uv_ref, const std::vector<Vec2> &pixel_uv_cur);
template <typename NNFeatureDescriptorType>
bool NNFeatureMatcher::InferenceSession(const std::vector<NNFeatureDescriptorType> &descriptors_ref,
                                        const std::vector<NNFeatureDescriptorType> &descriptors_cur,
                                        const std::vector<Vec2> &pixel_uv_ref,
                                        const std::vector<Vec2> &pixel_uv_cur) {
    // Validate session and input.
    RETURN_FALSE_IF(descriptors_ref.empty());
    RETURN_FALSE_IF(descriptors_ref.size() != descriptors_cur.size() ||
                    pixel_uv_ref.size() != pixel_uv_cur.size() ||
                    descriptors_ref.size() != pixel_uv_ref.size());
    RETURN_FALSE_IF(!session_);

    // Prepare input tensor matrices.
    input_tensor_matrices_[0].setZero(pixel_uv_ref.size(), pixel_uv_ref.front().rows());
    input_tensor_matrices_[1].setZero(pixel_uv_cur.size(), pixel_uv_cur.front().rows());
    input_tensor_matrices_[2].setZero(descriptors_ref.size(), descriptors_ref.front().rows());
    input_tensor_matrices_[3].setZero(descriptors_cur.size(), descriptors_cur.front().rows());
    for (uint32_t i = 0; i < pixel_uv_ref.size(); ++i) {
        input_tensor_matrices_[0].row(i) = pixel_uv_ref[i].transpose();
    }
    for (uint32_t i = 0; i < pixel_uv_cur.size(); ++i) {
        input_tensor_matrices_[1].row(i) = pixel_uv_cur[i].transpose();
    }
    for (uint32_t i = 0; i < descriptors_ref.size(); ++i) {
        input_tensor_matrices_[2].row(i) = descriptors_ref[i].transpose();
    }
    for (uint32_t i = 0; i < descriptors_cur.size(); ++i) {
        input_tensor_matrices_[3].row(i) = descriptors_cur[i].transpose();
    }

    // Convert input tensor matrices to tensors.
    for (uint32_t i = 0; i < input_tensor_matrices_.size(); ++i) {
        OnnxRuntime::ConvertMatrixToTensor(input_tensor_matrices_[i], memory_info_, input_tensors_[i]);
    }

    // Inference.
    run_options_.SetRunLogVerbosityLevel(ORT_LOGGING_LEVEL_WARNING);
    std::vector<const char *> input_names_ptr;
    std::vector<const char *> output_names_ptr;
    for (const auto &name: input_names_) {
        input_names_ptr.emplace_back(name.c_str());
    }
    for (const auto &name: output_names_) {
        output_names_ptr.emplace_back(name.c_str());
    }
    output_tensors_ = session_.Run(run_options_, input_names_ptr.data(), input_tensors_.data(), input_tensors_.size(), output_names_ptr.data(), output_names_ptr.size());

    return true;
}

template bool NNFeatureMatcher::Match<SuperpointDescriptorType>(const std::vector<SuperpointDescriptorType> &descriptors_ref,
    const std::vector<SuperpointDescriptorType> &descriptors_cur, const std::vector<Vec2> &pixel_uv_ref, const std::vector<Vec2> &pixel_uv_cur,
    std::vector<Vec2> &matched_pixel_uv_cur, std::vector<uint8_t> &status);
template bool NNFeatureMatcher::Match<DiskDescriptorType>(const std::vector<DiskDescriptorType> &descriptors_ref,
    const std::vector<DiskDescriptorType> &descriptors_cur, const std::vector<Vec2> &pixel_uv_ref, const std::vector<Vec2> &pixel_uv_cur,
    std::vector<Vec2> &matched_pixel_uv_cur, std::vector<uint8_t> &status);
template <typename NNFeatureDescriptorType>
bool NNFeatureMatcher::Match(const std::vector<NNFeatureDescriptorType> &descriptors_ref,
                             const std::vector<NNFeatureDescriptorType> &descriptors_cur,
                             const std::vector<Vec2> &pixel_uv_ref,
                             const std::vector<Vec2> &pixel_uv_cur,
                             std::vector<Vec2> &matched_pixel_uv_cur,
                             std::vector<uint8_t> &status) {
    RETURN_FALSE_IF_FALSE(InferenceSession(descriptors_ref, descriptors_cur, pixel_uv_ref, pixel_uv_cur));

    // Convert output tensors to image matrices.
    status.resize(pixel_uv_ref.size(), static_cast<uint8_t>(TrackStatus::kNotTracked));
    if (output_tensors_.size() == 2) {
        // Convert output tensors to image matrices.
        std::vector<Eigen::Map<const TMatImg<int64_t>>> matches_matrices;
        OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[0], matches_matrices);
        std::vector<Eigen::Map<const TMatImg<float>>> descriptors_matrices;
        OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[1], descriptors_matrices);

        // Post process matches matrix.
        matched_pixel_uv_cur.clear();
        for (uint32_t i = 0; i < matches_matrices[0].rows(); ++i) {
            matched_pixel_uv_cur.emplace_back(pixel_uv_cur[matches_matrices[0](i, 1)]);
            status[matches_matrices[0](i, 0)] = static_cast<uint8_t>(TrackStatus::kTracked);
        }

    } else {
        std::vector<Eigen::Map<const TMatImg<float>>> scores_matrices;
        OnnxRuntime::ConvertTensorToImageMatrice(output_tensors_[0], scores_matrices);
        const auto &scores_matrix = scores_matrices[0];

        // Post process scores matrix:
        // Find the max score with index for each row.
        // Find the max score with index for each column.
        // Check if the max score in each row is the same as the max score in each column. If not, set the score to 0.
        // Finally score of each pair should be exp(score). Which is during [0, 1].
        // But there is no need to compute the exp(score), because compare score is enough.

        std::vector<uint32_t> max_scores_in_cols_index;
        for (uint32_t j = 0; j < scores_matrix.cols(); ++j) {
            const auto &scores = scores_matrix.col(j);
            uint32_t max_score_index = 0;
            float max_score = scores(max_score_index);
            for (uint32_t i = 1; i < scores.rows(); ++i) {
                if (scores(i) > max_score) {
                    max_score = scores(i);
                    max_score_index = i;
                }
            }
            max_scores_in_cols_index.emplace_back(max_score_index);
        }

        matched_pixel_uv_cur.clear();
        for (uint32_t i = 0; i < scores_matrix.rows(); ++i) {
            if (scores_matrix(i, max_scores_in_cols_index[i]) > options_.kMinValidMatchScore) {
                matched_pixel_uv_cur.emplace_back(pixel_uv_cur[i]);
                status[i] = static_cast<uint8_t>(TrackStatus::kTracked);
            }
        }
    }

    return true;
}



} // namespace FEATURE_TRACKER
