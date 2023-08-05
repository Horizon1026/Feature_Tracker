#include "optical_flow_lssd_klt.h"
#include "slam_memory.h"
#include "log_report.h"
#include "slam_operations.h"

namespace FEATURE_TRACKER {

void OpticalFlowLssdKlt::TrackOneFeatureFast(const GrayImage &ref_image,
                                             const GrayImage &cur_image,
                                             const Vec2 &ref_pixel_uv,
                                             Mat2 &R_cr,
                                             Vec2 &t_cr,
                                             uint8_t &status) {
    // Confirm extended patch size. Extract it from reference image.
    ex_ref_patch().clear();
    ex_ref_patch_pixel_valid().clear();
    const uint32_t valid_pixel_num = ExtractExtendPatchInReferenceImage(ref_image, ref_pixel_uv, ex_ref_patch_rows(), ex_ref_patch_cols(), ex_ref_patch(), ex_ref_patch_pixel_valid());

    // If this feature has no valid pixel in patch, it can not be tracked.
    if (valid_pixel_num == 0) {
        status = static_cast<uint8_t>(TrackStatus::kOutside);
        return;
    }

}

}
