#ifndef _KLT_DATATYPE_H_
#define _KLT_DATATYPE_H_

#include <cmath>

namespace KLT_TRACKER {

using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;

using int8_t = char;
using int16_t = short;
using int32_t = int;

inline constexpr static uint32_t kPyramidMaxLevel = 10;

class Image {
public:
    explicit Image() = default;
    virtual ~Image() = default;
    explicit Image(uint8_t *data, int32_t rows, int32_t cols);

    uint8_t *image_data() const { return image_data_; }
    int32_t cols() const { return cols_; }
    int32_t rows() const { return rows_; }

    bool SetImage(uint8_t *data, int32_t rows, int32_t cols);
    void SetSize(int32_t rows, int32_t cols);

    bool SetPixelValue(int32_t row, int32_t col, uint8_t value) {
        if (col < 0 || row < 0 || col > cols_ - 1 || row > rows_ - 1) {
            return false;
        }

        image_data_[row * cols_ + col] = value;
        return true;
    }

    bool GetPixelValue(int32_t row, int32_t col, uint16_t *value) const {
        if (col < 0 || row < 0 || col > cols_ - 1 || row > rows_ - 1) {
            return false;
        }

        *value = static_cast<uint16_t>(image_data_[row * cols_ + col]);
        return true;
    }

    bool GetPixelValue(float row, float col, float *value) const {
        if (col < 0 || row < 0 || col > cols_ - 1 || row > rows_ - 1) {
            return false;
        }

        uint8_t *values = &image_data_[static_cast<int32_t>(row) * cols_ + static_cast<int32_t>(col)];
        float sub_row = row - std::floor(row);
        float sub_col = col - std::floor(col);
        float inv_sub_row = 1.0f - sub_row;
        float inv_sub_col = 1.0f - sub_col;

        *value = static_cast<float>(
            inv_sub_col * inv_sub_row * values[0] + sub_col * inv_sub_row * values[1] +
            inv_sub_col * sub_row * values[cols_] + sub_col * sub_row * values[cols_ + 1]);

        return true;
    }

private:
    uint8_t *image_data_ = nullptr;
    int32_t cols_ = 0;
    int32_t rows_ = 0;
};

class ImagePyramid {
public:
    explicit ImagePyramid() = default;
    virtual ~ImagePyramid() = default;
    explicit ImagePyramid(uint32_t level, Image *raw_image);
    explicit ImagePyramid(uint32_t level, uint8_t *image_data, int32_t rows, int32_t cols);

    uint32_t level() const { return level_; }
    Image *images() { return images_; }
    uint8_t *pyramid_buf() const { return pyramid_buf_; }

    Image GetImage(uint32_t level_idx) const { return images_[level_idx]; }

    bool SetRawImage(Image *raw_image);
    bool SetRawImage(uint8_t *image_data, int32_t rows, int32_t cols);
    bool SetPyramidBuff(uint8_t *pyramid_buf);

    bool CreateImagePyramid(uint32_t level);

private:
    uint32_t level_ = 0;
    Image images_[kPyramidMaxLevel];
    uint8_t *pyramid_buf_ = nullptr;
};
}

#endif
