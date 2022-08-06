#include <FASTFeatureDetector.h>

/* 不带参数的构造函数 */
FASTFeatureDetectorClass::FASTFeatureDetectorClass() {
    // 设定默认参数
    threshold = 0.2;        // FAST 角点检测阈值，范围 0 - 1
    edgeDistance = 10;      // FAST 角点检测屏蔽边框大小，靠近图像边缘的不检测
    N = 12;                 // 若一个像素点周围有 N 个像素点满足要求，则认为这个像素点是 FAST 角点
    minDistance = 15;       // 检测稀疏角点时，两个特征点之间的最小距离
    minGrayValueDiff = 10;  // FAST 角点检测最小灰度值差
}


/* 带参数的构造函数 */
FASTFeatureDetectorClass::FASTFeatureDetectorClass(float SET_threshold,
                                                   int SET_edgeDistance,
                                                   int SET_N,
                                                   int SET_minDistance,
                                                   int SET_minGrayValueDiff) {
    threshold = SET_threshold;          // FAST 角点检测阈值，范围 0 - 1
    edgeDistance = SET_edgeDistance;    // FAST 角点检测屏蔽边框大小，靠近图像边缘的不检测
    N = SET_N;                          // 若一个像素点周围有 N 个像素点满足要求，则认为这个像素点是 FAST 角点
    minDistance = SET_minDistance;      // 检测稀疏角点时，两个特征点之间的最小距离
    minGrayValueDiff = SET_minGrayValueDiff;    // FAST 角点检测最小灰度值差
    // 如果检测阈值不在范围内，则强行调整
    if (threshold < 0 || threshold > 1) {
        threshold = 0.2;
    }
    // 如果屏蔽边框过小，则强制赋值
    if (edgeDistance < 3) {
        edgeDistance = 3;
    }
    // 如果最少满足数 N 不在合理范围内，则强行调整
    if (N < 1 || N > 15) {
        N = 12;
    }
    // 如果稀疏点检测的特征点之间的最小距离不再合理范围内，则强行调整
    if (minDistance < 1) {
        minDistance = 1;
    }
    // 如果最小灰度值差过小，则强制赋值
    if (minGrayValueDiff < 0) {
        minGrayValueDiff = 0;
    }
}

/* 判断一幅图像中的某一个像素点是否为 FAST 角点，默认输入信息全部合法 */
inline bool FASTFeatureDetectorClass::isFASTFeature(cv::Mat &image, int u, int v) {
    int score = 0;
    return FASTFeatureDetectorClass::isFASTFeature(image, u, v, score);
}


/* 判断一幅图像中的某一个像素点是否为 FAST 角点，并返回这个角点的得分，默认输入信息全部合法 */
inline bool FASTFeatureDetectorClass::isFASTFeature(cv::Mat &image, int u, int v, int &score) {
    // 得分默认为 0
    score = 0;

    // 根据阈值计算当前像素点的判断上下限
    int midGrayValue = image.at<uchar>(v, u);
    int diff = minGrayValueDiff > int(threshold * float(midGrayValue)) ? minGrayValueDiff : int(threshold * float(midGrayValue));
    int maxThreshold = midGrayValue + diff;
    int minThreshold = midGrayValue - diff;
    int maxCount = 0;
    int minCount = 0;

    // 如果是FAST-12及其以上算法，则可以预测试
    if (N >= 12) {
        std::vector<unsigned int> subIndex = {0, 4, 8, 12};
        for (unsigned int i = 0; i < subIndex.size(); i++) {
            int grayValue = image.at<uchar>(v + index[subIndex[i]].second, u + index[subIndex[i]].first);
            if (grayValue > maxThreshold) {
                maxCount++;
                minCount = 0;
            } else if (grayValue < minThreshold) {
                minCount++;
                maxCount = 0;
            } else {
                minCount = 0;
                maxCount = 0;
            }
        }
        // 如果不存在连续 3 个同时大于最大阈值或者同时小于最小阈值的情况，则不可能是角点
        if (minCount < 3 && maxCount < 3) {
            return false;
        }
    }

    // 构造比较结果序列，小于最小阈值为-1，大于最大阈值为+1
    std::vector<int> compareResult(index.size(), 0);
    for (unsigned int i = 0; i < index.size(); i++) {
        int grayValue = image.at<uchar>(v + index[i].second, u + index[i].first);
        if (grayValue > maxThreshold) {
            compareResult[i] = 1;
        } else if (grayValue < minThreshold) {
            compareResult[i] = -1;
        }
    }

    // 遍历两次比较结果序列进行计数
    minCount = 0;
    maxCount = 0;
    for (int k = 0; k < 2; k++) {
        for (unsigned int i = 0; i < compareResult.size(); i++) {
            if (compareResult[i] == 1) {
                maxCount++;
                minCount = 0;
            } else if (compareResult[i] == -1) {
                minCount++;
                maxCount = 0;
            } else {
                minCount = 0;
                maxCount = 0;
            }
            // 如果连续 N 个大于最大阈值，或者连续 N 个小于最小阈值，则此点为 FAST 角点
            if (maxCount >= N || minCount >= N) {
                score = maxCount > minCount ? maxCount : minCount;
                return true;
            }
        }
    }
    return false;
}


/* 检测一幅图像中的所有 FAST 角点 */
std::vector<cv::Point2f> FASTFeatureDetectorClass::DetectAllFeatures(cv::Mat &image) {
    std::vector<cv::Point2f> points;

    // 如果图像是空的，则返回空列表
    if (image.data == nullptr) {
        return points;
    }

    // 获取图像的尺寸
    int ROW = image.rows;
    int COL = image.cols;

    // 遍历整个图像区域
    for (int u = edgeDistance; u < COL - edgeDistance; u++) {
        for (int v = edgeDistance; v < ROW - edgeDistance; v++) {
            if (isFASTFeature(image, u, v)) {
                points.emplace_back(cv::Point2f(u, v));
            }
        }
    }

    // 返回检测到的所有特征点
    return points;
}


/* 检测一幅图像中的稀疏的 FAST 角点 */
std::vector<cv::Point2f> FASTFeatureDetectorClass::DetectSparseFeatures(cv::Mat &image) {
    std::vector<cv::Point2f> points;

    // 如果图像是空的，则返回空列表
    if (image.data == nullptr) {
        return points;
    }

    // 获取图像的尺寸
    int ROW = image.rows;
    int COL = image.cols;

    // 构造屏蔽域
    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    // 遍历整个图像区域
    for (int u = edgeDistance; u < COL - edgeDistance; u++) {
        for (int v = edgeDistance; v < ROW - edgeDistance; v++) {
            // 如果当前遍历到的点已经被屏蔽，则继续下一个
            if (mask.at<uchar>(v, u) == 0) {
                continue;
            }
            // 否则判断当前点是否为 FAST 角点，是的话就添加到列表中，并更新屏蔽域
            if (isFASTFeature(image, u, v)) {
                points.emplace_back(cv::Point2f(u, v));
                cv::circle(mask, cv::Point2f(u, v), minDistance, 0, -1);
            }
        }
    }

    // 返回检测到的所有特征点
    return points;
}


/* 在给定屏蔽域的情况下，检测一幅图像中的稀疏的 FAST 角点 */
std::vector<cv::Point2f> FASTFeatureDetectorClass::DetectSparseFeatures(cv::Mat &image, cv::Mat &mask) {
    std::vector<cv::Point2f> points;

    // 如果图像是空的，则返回空列表
    if (image.data == nullptr) {
        return points;
    }

    // 如果屏蔽域和原图像尺寸不一样，返回空列表
    if (image.cols != mask.cols || image.rows != mask.rows) {
        return points;
    }

    // 遍历整个图像区域
    for (int u = edgeDistance; u < image.cols - edgeDistance; u++) {
        for (int v = edgeDistance; v < image.rows - edgeDistance; v++) {
            // 如果当前遍历到的点已经被屏蔽，则继续下一个
            if (mask.at<uchar>(v, u) == 0) {
                continue;
            }
            // 否则判断当前点是否为 FAST 角点，是的话就添加到列表中，并更新屏蔽域
            if (isFASTFeature(image, u, v)) {
                points.emplace_back(cv::Point2f(u, v));
                cv::circle(mask, cv::Point2f(u, v), minDistance, 0, -1);
            }
        }
    }

    // 返回检测到的所有特征点
    return points;
}


/* 检测一幅图像中的评价较高的稀疏的 FAST 角点 */
std::vector<cv::Point2f> FASTFeatureDetectorClass::DetectGoodSparseFeatures(cv::Mat &image) {
    std::vector<cv::Point2f> points;

    // 每一种评分（16 -> N）都单独排一行，最高分（16）的脚标为 0
    std::vector<std::vector<cv::Point2f>> allPoints(16 - N + 1);

    // 如果图像是空的，则返回空列表
    if (image.data == nullptr) {
        return points;
    }

    // 遍历整个图像区域，检测所有的 FAST 角点并根据排名进行桶排序
    for (int u = edgeDistance; u < image.cols - edgeDistance; u++) {
        for (int v = edgeDistance; v < image.rows - edgeDistance; v++) {
            int score = 0;
            if (isFASTFeature(image, u, v, score)) {
                int idx = 16 - score;
                allPoints[idx].emplace_back(cv::Point2f(u, v));
            }
        }
    }

    // 构造屏蔽域
    cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    // 从最高分到最低分依次遍历所有的 FAST 角点
    for (unsigned int i = 0; i < allPoints.size(); i++) {
        for (unsigned int j = 0; j < allPoints[i].size(); j++) {
            // 如果当前角点的像素坐标没有被屏蔽，则将此点放到输出列表中，并更新屏蔽域
            if (mask.at<uchar>(allPoints[i][j].y, allPoints[i][j].x) == 255) {
                points.emplace_back(allPoints[i][j]);
                cv::circle(mask, allPoints[i][j], minDistance, 0, -1);
            }
        }
    }

    // 返回检测到的所有特征点
    return points;
}


/* 在给定屏蔽域的情况下，检测一幅图像中的评价较高的稀疏的 FAST 角点 */
std::vector<cv::Point2f> FASTFeatureDetectorClass::DetectGoodSparseFeatures(cv::Mat &image, cv::Mat &mask) {
    std::vector<cv::Point2f> points;

    // 每一种评分（16 -> N）都单独排一行，最高分（16）的脚标为 0
    std::vector<std::vector<cv::Point2f>> allPoints(16 - N + 1);

    // 如果图像是空的，则返回空列表
    if (image.data == nullptr) {
        return points;
    }

    // 如果屏蔽域和原图像尺寸不一样，返回空列表
    if (image.cols != mask.cols || image.rows != mask.rows) {
        return points;
    }

    // 遍历整个图像区域，检测所有的 FAST 角点并根据排名进行桶排序
    for (int u = edgeDistance; u < image.cols - edgeDistance; u++) {
        for (int v = edgeDistance; v < image.rows - edgeDistance; v++) {
            int score = 0;
            if (isFASTFeature(image, u, v, score)) {
                int idx = 16 - score;
                allPoints[idx].emplace_back(cv::Point2f(u, v));
            }
        }
    }

    // 从最高分到最低分依次遍历所有的 FAST 角点
    for (unsigned int i = 0; i < allPoints.size(); i++) {
        for (unsigned int j = 0; j < allPoints[i].size(); j++) {
            // 如果当前角点的像素坐标没有被屏蔽，则将此点放到输出列表中，并更新屏蔽域
            if (mask.at<uchar>(allPoints[i][j].y, allPoints[i][j].x) == 255) {
                points.emplace_back(allPoints[i][j]);
                cv::circle(mask, allPoints[i][j], minDistance, 0, -1);
            }
        }
    }

    // 返回检测到的所有特征点
    return points;
}