#include "optical_flow_basic_klt.h"
#include "simd_wrapper.h"

namespace feature_tracker {

using namespace slam_utility::simd;

int32_t OpticalFlowBasicKlt::ConstructIncrementalFunctionSIMD(const GrayImage &ref_image, const GrayImage &cur_image, 
                                                              const Vec2 &ref_pixel_uv, const Vec2 &cur_pixel_uv, 
                                                              Mat2 &hessian, Vec2 &bias) {
    const int32_t half_size = options().kPatchRowHalfSize;
    int32_t num_of_valid_pixel = 0;

    // SIMD accumulators
    float32x4_t h00_vec = SetF(0.0f);
    float32x4_t h11_vec = SetF(0.0f);
    float32x4_t h01_vec = SetF(0.0f);
    float32x4_t b0_vec = SetF(0.0f);
    float32x4_t b1_vec = SetF(0.0f);

    // Temp buffers for batch processing (size 4)
    float fx_buf[4], fy_buf[4], ft_buf[4];
    int32_t buf_ptr = 0;

    for (int32_t drow = -half_size; drow <= half_size; ++drow) {
        for (int32_t dcol = -half_size; dcol <= half_size; ++dcol) {
            const float row_i = static_cast<float>(drow) + ref_pixel_uv.y();
            const float col_i = static_cast<float>(dcol) + ref_pixel_uv.x();
            const float row_j = static_cast<float>(drow) + cur_pixel_uv.y();
            const float col_j = static_cast<float>(dcol) + cur_pixel_uv.x();

            float v0, v1, v2, v3, v4, v5;
            if (ref_image.GetPixelValue(row_i, col_i - 1.0f, &v0) && ref_image.GetPixelValue(row_i, col_i + 1.0f, &v1) &&
                ref_image.GetPixelValue(row_i - 1.0f, col_i, &v2) && ref_image.GetPixelValue(row_i + 1.0f, col_i, &v3) &&
                ref_image.GetPixelValue(row_i, col_i, &v4) && cur_image.GetPixelValue(row_j, col_j, &v5)) {
                
                fx_buf[buf_ptr] = v1 - v0;
                fy_buf[buf_ptr] = v3 - v2;
                ft_buf[buf_ptr] = v5 - v4;
                buf_ptr++;
                num_of_valid_pixel++;

                if (buf_ptr == 4) {
                    float32x4_t fx = LoadF(fx_buf);
                    float32x4_t fy = LoadF(fy_buf);
                    float32x4_t ft = LoadF(ft_buf);

                    h00_vec = AddF(h00_vec, MulF(fx, fx));
                    h11_vec = AddF(h11_vec, MulF(fy, fy));
                    h01_vec = AddF(h01_vec, MulF(fx, fy));
                    b0_vec = SubF(b0_vec, MulF(fx, ft));
                    b1_vec = SubF(b1_vec, MulF(fy, ft));
                    
                    buf_ptr = 0;
                }
            }
        }
    }

    // Process remainder in buffer
    for (int32_t i = 0; i < buf_ptr; ++i) {
        float fx = fx_buf[i];
        float fy = fy_buf[i];
        float ft = ft_buf[i];
        hessian(0, 0) += fx * fx;
        hessian(1, 1) += fy * fy;
        hessian(0, 1) += fx * fy;
        bias(0) -= fx * ft;
        bias(1) -= fy * ft;
    }
    
    // Reduce SIMD results
    hessian(0, 0) += ReduceSumF(h00_vec);
    hessian(1, 1) += ReduceSumF(h11_vec);
    hessian(0, 1) += ReduceSumF(h01_vec);
    hessian(1, 0) = hessian(0, 1);
    bias(0) += ReduceSumF(b0_vec);
    bias(1) += ReduceSumF(b1_vec);

    return num_of_valid_pixel;
}

}
