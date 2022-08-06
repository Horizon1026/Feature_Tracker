#pragma once

#include <opencv2/opencv.hpp>

class FASTFeatureDetectorClass {
private:
    // FAST 角点检测阈值，范围 0 - 1
    float threshold;
    // FAST 角点检测屏蔽边框大小，靠近图像边缘的不检测
    int edgeDistance;
    // 若一个像素点周围有 N 个像素点满足要求，则认为这个像素点是 FAST 角点
    int N;
    // FAST 角点周围像素点的相对坐标，pair<du, dv>
    std::vector<std::pair<int, int>> index = {
        {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
        {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3} };
    // 检测稀疏角点时，两个特征点之间的最小距离
    int minDistance;
    // FAST 角点检测最小灰度值差
    int minGrayValueDiff;

public:
    /* 不带参数的构造函数 */
    FASTFeatureDetectorClass();
    /* 带参数的构造函数 */
    FASTFeatureDetectorClass(float SET_threshold,
                             int SET_edgeDistance,
                             int SET_N,
                             int SET_minDistance,
                             int SET_minGrayValueDiff);

private:
    /* 判断一幅图像中的某一个像素点是否为 FAST 角点，默认输入信息全部合法 */
    inline bool isFASTFeature(cv::Mat &image, int u, int v);

    /* 判断一幅图像中的某一个像素点是否为 FAST 角点，并返回这个角点的得分，默认输入信息全部合法 */
    inline bool isFASTFeature(cv::Mat &image, int u, int v, int &score);

public:
    /* 检测一幅图像中的所有 FAST 角点 */
    std::vector<cv::Point2f> DetectAllFeatures(cv::Mat &image);

    /* 检测一幅图像中的稀疏的 FAST 角点 */
    std::vector<cv::Point2f> DetectSparseFeatures(cv::Mat &image);

    /* 在给定屏蔽域的情况下，检测一幅图像中的稀疏的 FAST 角点 */
    std::vector<cv::Point2f> DetectSparseFeatures(cv::Mat &image, cv::Mat &mask);

    /* 检测一幅图像中的评价较高的稀疏的 FAST 角点 */
    std::vector<cv::Point2f> DetectGoodSparseFeatures(cv::Mat &image);

    /* 在给定屏蔽域的情况下，检测一幅图像中的评价较高的稀疏的 FAST 角点 */
    std::vector<cv::Point2f> DetectGoodSparseFeatures(cv::Mat &image, cv::Mat &mask);
};