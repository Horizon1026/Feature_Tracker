#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class OpticalFlowClass {
private:
    int levelsNum;          // 光流金字塔的层数
    int halfBatchSize;      // 像素块的半径
    double convergeThreshold = 1;     //收敛判断阈值设置为 1

public:
    static const uchar METHOD_LSE_FORWORD = 0;
    static const uchar METHOD_LSE_INVERSE = 1;
    static const uchar METHOD_GN_FORWORD = 2;
    static const uchar METHOD_GN_INVERSE = 3;

public:
    /* 不带参数的构造函数 */
    OpticalFlowClass();
    /* 带参数的构造函数 */
    OpticalFlowClass(int SET_levelsNum, int SET_halfBatchSize);

private:
    /* 采用双线性插值的方法从图像中提取灰度值 */
    inline float GetPixelValue(const cv::Mat &matirx, float x, float y);
    /* 计算图像的积分图 */
    inline cv::Mat ComputeIntegralMatrix(const cv::Mat &matrix);
    /* 计算图像的微分图 */
    inline void ComputeDerivateMatrix(const cv::Mat &matrix, cv::Mat &Ix, cv::Mat &Iy);
    /* 根据当前像素点坐标，确定以此为中心的图像块的范围 */
    inline void ComputeBatchRange(const cv::Point2f &point, const int rows, const int cols , float &sx, float &ex, float &sy, float &ey);

private:
    /* 单层正向光流追踪特征点，采用最小二乘方法 */
    void TrackFeaturesSingleLevel_LSE(const cv::Mat &image0,
                                      const cv::Mat &image1,
                                      const std::vector<cv::Point2f> &points0,
                                      std::vector<cv::Point2f> &points1,
                                      std::vector<uchar> &status);
    /* 单层反向光流追踪特征点，采用最小二乘方法 */
    void TrackFeaturesSingleLevel_inverse_LSE(const cv::Mat &image0,
                                              const cv::Mat &image1,
                                              const std::vector<cv::Point2f> &points0,
                                              std::vector<cv::Point2f> &points1,
                                              std::vector<uchar> &status);
    /* 单层正向光流追踪特征点，采用高斯牛顿方法 */
    void TrackFeaturesSingleLevel_GN(const cv::Mat &image0,
                                     const cv::Mat &image1,
                                     const std::vector<cv::Point2f> &points0,
                                     std::vector<cv::Point2f> &points1,
                                     std::vector<uchar> &status);
    /* 单层反向光流追踪特征点，采用高斯牛顿方法 */
    void TrackFeaturesSingleLevel_inverse_GN(const cv::Mat &image0,
                                             const cv::Mat &image1,
                                             const std::vector<cv::Point2f> &points0,
                                             std::vector<cv::Point2f> &points1,
                                             std::vector<uchar> &status);

public:
    /* 多层光流金字塔追踪特征点 */
    void TrackFeaturesMultiLevels(const cv::Mat &image0,
                                  const cv::Mat &image1,
                                  const std::vector<cv::Point2f> &points0,
                                  std::vector<cv::Point2f> &points1,
                                  std::vector<uchar> &status,
                                  uchar method = 1);    // 默认使用反向的 LSE 方法
};