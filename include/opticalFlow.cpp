#include <opticalFlow.h>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// TODO:
#include <ctime>
clock_t sTime, eTime;

/* 不带参数的构造函数 */
OpticalFlowClass::OpticalFlowClass() {
    levelsNum = 4;
    halfBatchSize = 10;
}


/* 带参数的构造函数 */
OpticalFlowClass::OpticalFlowClass(int SET_levelsNum, int SET_halfBatchSize) {
    // 如果设置的金字塔层数符合实际则直接赋值，否则使用默认值
    if (SET_levelsNum > 0 && SET_levelsNum < 6) {
        levelsNum = SET_levelsNum;
    } else {
        levelsNum = 4;
    }
    // 如果设置的图像块半径符合实际则直接赋值，否则使用默认值
    if (SET_halfBatchSize > -1) {
        halfBatchSize = SET_halfBatchSize;
    } else {
        halfBatchSize = 10;
    }
}


/* 采用双线性插值的方法从图像中提取灰度值 */
inline float OpticalFlowClass::GetPixelValue(const cv::Mat &matrix, float x, float y) {
    // 检查是否存在数组越界
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > matrix.cols - 1) x = matrix.cols - 1;
    if (y > matrix.rows - 1) y = matrix.rows - 1;

    // 双线性插值返回结果
    if (matrix.type() == CV_8UC1) {
        uchar *data = &matrix.data[int(y) * matrix.step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[matrix.step] +
            xx * yy * data[matrix.step + 1]
        );
    } else if (matrix.type() == CV_32FC1) {
        int ix = floor(x);
        int iy = floor(y);
        float xx = x - ix;
        float yy = y - iy;
        return float(
            (1 - xx) * (1 - yy) * matrix.at<float>(iy, ix) +
            xx * (1 - yy) * matrix.at<float>(iy, ix + 1) +
            (1 - xx) * yy * matrix.at<float>(iy + 1, ix) +
            xx * yy * matrix.at<float>(iy + 1, ix + 1)
        );
    }
        
}


/* 计算图像的积分图 */
inline cv::Mat OpticalFlowClass::ComputeIntegralMatrix(const cv::Mat &matrix) {
    // 将 uchar 类型的图像转化成 float 类型
    cv::Mat integralMatrix;
    matrix.convertTo(integralMatrix, CV_32FC1, 1.0);
    
    // 计算边界值
    for (unsigned int u = 1; u < integralMatrix.cols; u++) {
        integralMatrix.at<float>(0, u) += integralMatrix.at<float>(0, u - 1);
    }
    for (unsigned int v = 1; v < integralMatrix.rows; v++) {
        integralMatrix.at<float>(v, 0) += integralMatrix.at<float>(v - 1, 0);
    }

    // 计算中间区域值
    for (unsigned int u = 1; u < integralMatrix.cols; u++) {
        for (unsigned int v = 1; v < integralMatrix.rows; v++) {
            integralMatrix.at<float>(v, u) = integralMatrix.at<float>(v - 1, u) + integralMatrix.at<float>(v, u - 1) - integralMatrix.at<float>(v - 1, u - 1);
        }
    }

    // 返回结果
    return integralMatrix;
}


/* 计算图像的微分图 */
inline void OpticalFlowClass::ComputeDerivateMatrix(const cv::Mat &matrix, cv::Mat &Ix, cv::Mat &Iy) {
    /* 采用 Scharr 微分算子 */
    /*
    -3   0  3          -3  -10  -3
    -10  0  10          0    0   0
    -3   0  3           3   10   3
    */

    Ix = cv::Mat(matrix.rows, matrix.cols, CV_32FC1);
    Iy = cv::Mat(matrix.rows, matrix.cols, CV_32FC1);

    // 边缘值全部设置为 0
    for (unsigned int u = 0; u < matrix.cols; u++) {
        Ix.at<float>(0, u) = 0;
        Iy.at<float>(0, u) = 0;
    }
    for (unsigned int v = 0; v < matrix.rows; v++) {
        Ix.at<float>(v, 0) = 0;
        Iy.at<float>(v, 0) = 0;
    }

    // 由算子计算中间的梯度值
    for (int u = 1; u < matrix.cols - 1; u++) {
        for (int v = 1; v < matrix.rows - 1; v++) {
            Ix.at<float>(v, u) = - matrix.at<uchar>(v - 1, u - 1) * 3
                                 - matrix.at<uchar>(v,     u - 1) * 10
                                 - matrix.at<uchar>(v + 1, u - 1) * 3
                                 + matrix.at<uchar>(v - 1, u + 1) * 3
                                 + matrix.at<uchar>(v,     u + 1) * 10
                                 + matrix.at<uchar>(v + 1, u + 1) * 3;
            Iy.at<float>(v, u) = - matrix.at<uchar>(v - 1, u - 1) * 3
                                 - matrix.at<uchar>(v - 1, u    ) * 10
                                 - matrix.at<uchar>(v - 1, u + 1) * 3
                                 + matrix.at<uchar>(v + 1, u - 1) * 3
                                 + matrix.at<uchar>(v + 1, u    ) * 10
                                 + matrix.at<uchar>(v + 1, u + 1) * 3;
            Ix.at<float>(v, u) /= 32.0;
            Iy.at<float>(v, u) /= 32.0;
        }
    }
}


/* 根据当前像素点坐标，确定以此为中心的图像块的范围 */
inline void OpticalFlowClass::ComputeBatchRange(const cv::Point2f &point, const int rows, const int cols , float &sx, float &ex, float &sy, float &ey) {
    sx = point.x - float(halfBatchSize);
    if (floor(sx) < 1) {
        sx += 1 - floor(sx);
    }
    sy = point.y - float(halfBatchSize);
    if (floor(sy) < 1) {
        sy += 1 - floor(sy);
    }
    ex = point.x + float(halfBatchSize);
    if (floor(ex) > cols - 3) {
        ex -= floor(ex) - cols + 3;
    }
    ey = point.y + float(halfBatchSize);
    if (floor(ey) > rows - 3) {
        ey -= floor(ey) - rows + 3;
    }
}


/* 单层正向光流追踪特征点，采用最小二乘方法 */
void OpticalFlowClass::TrackFeaturesSingleLevel_LSE(const cv::Mat &image0,
                                                    const cv::Mat &image1,
                                                    const std::vector<cv::Point2f> &points0,
                                                    std::vector<cv::Point2f> &points1,
                                                    std::vector<uchar> &status) {
    // 如果没有设定初值，则调整初值
    if (points1.size() != points0.size() || status.size() != points0.size()) {
        points1 = points0;
        status.resize(points0.size(), 1);
    }

    // 计算图像 1 的微分图，原图耗时 2 ms
    cv::Mat Ix, Iy;
    ComputeDerivateMatrix(image1, Ix, Iy);

    // 遍历图像 0 中的每一个特征点
    for (unsigned int i = 0; i < points0.size(); i++) {
        // 定义引用
        const cv::Point2f &point0 = points0[i];
        cv::Point2f &point1 = points1[i];
        uchar &matchResult = status[i];

        // 如果当前这个特征点已经追踪失败，则不再考虑
        if (matchResult == 0) {
            continue;
        }

        // 如果当前点已经追踪到了图像边界之外，则认为追踪失败不再考虑
        if (point1.x < 0 || point1.x > image1.cols - 1 ||
            point1.y < 0 || point1.y > image1.rows - 1) {
            matchResult = 0;
            continue;
        }

        // 定义最大迭代次数
        int maxEpoch = 50;
        while (maxEpoch > 0) {
            maxEpoch--;

            // 构造 G 矩阵，G = (A_T * A)_-1 * A_T
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();

            // 遍历像素块内的每一个像素，计算出 H * v = b 方程中的 b
            for (int dx = - halfBatchSize; dx <= halfBatchSize; dx++) {
                for (int dy = - halfBatchSize; dy <= halfBatchSize; dy++) {
                    double Ixi = GetPixelValue(Ix, point1.x + dx, point1.y + dy);
                    double Iyi = GetPixelValue(Iy, point1.x + dx, point1.y + dy);
                    double Iti = GetPixelValue(image1, point1.x + dx, point1.y + dy) - GetPixelValue(image0, point0.x + dx, point0.y + dy);
                    H(0, 0) += Ixi * Ixi;
                    H(0, 1) += Ixi * Iyi;
                    H(1, 1) += Iyi * Iyi;
                    b(0, 0) -= Ixi * Iti;
                    b(1, 0) -= Iyi * Iti;
                }
            }
            H(1, 0) = H(0, 1);

            // 计算出光流速度，更新追踪位置
            Eigen::Vector2d v = H.ldlt().solve(b);

            // 如果求解失败，则认为此点无法追踪
            if (std::isnan(v(0, 0)) || std::isnan(v(1, 0))) {
                matchResult = 0;
                break;
            }

            // 更新位置
            point1.x += v(0, 0);
            point1.y += v(1, 0);

            // 如果更新之后的位置超出了图像范围，则认为跟踪失败
            if (point1.x < 0 || point1.x > image1.cols || point1.y < 0 || point1.y > image1.rows) {
                matchResult = 0;
                break;
            }

            // 如果调整量非常小，则认为收敛
            if (v.norm() < convergeThreshold) {
                matchResult = 1;
                break;
            }
        }
    }
}


/* 单层反向光流追踪特征点，采用最小二乘方法 */
void OpticalFlowClass::TrackFeaturesSingleLevel_inverse_LSE(const cv::Mat &image0,
                                                            const cv::Mat &image1,
                                                            const std::vector<cv::Point2f> &points0,
                                                            std::vector<cv::Point2f> &points1,
                                                            std::vector<uchar> &status) {
    // 如果没有设定初值，则调整初值
    if (points1.size() != points0.size() || status.size() != points0.size()) {
        points1 = points0;
        status.resize(points0.size(), 1);
    }

    // 计算图像 0 的微分图， 480 * 750 的图耗时 2ms
    cv::Mat Ix, Iy;
    ComputeDerivateMatrix(image0, Ix, Iy);

    // 计算图像 0 中每一个特征点对应的 H 矩阵，Ixi 和 Iyi，特征点数 250 时固定耗时 2.2ms
    std::vector<Eigen::Matrix2d> all_H(points0.size());
    std::vector<std::vector<double>> all_Ixi(points0.size()), all_Iyi(points0.size());
    for (unsigned int i = 0; i < points0.size(); i++) {
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        int batchSize = 2 * halfBatchSize + 1;
        std::vector<double> Ixis(batchSize * batchSize), Iyis(batchSize * batchSize);
        // 遍历像素块内的每一个像素，计算出 H * v = b 方程中的 H
        int idx = 0;
        for (int dx = - halfBatchSize; dx <= halfBatchSize; dx++) {
            for (int dy = - halfBatchSize; dy <= halfBatchSize; dy++) {
                double Ixi = GetPixelValue(Ix, points0[i].x + dx, points0[i].y + dy);
                double Iyi = GetPixelValue(Iy, points0[i].x + dx, points0[i].y + dy);
                H(0, 0) += Ixi * Ixi;
                H(0, 1) += Ixi * Iyi;
                H(1, 1) += Iyi * Iyi;
                Ixis[idx] = Ixi;
                Iyis[idx] = Iyi;
                idx++;
            }
        }
        H(1, 0) = H(0, 1);
        all_H[i] = H;
        all_Ixi[i] = Ixis;
        all_Iyi[i] = Iyis;
    }

    // 遍历图像 0 中的每一个特征点
    for (unsigned int i = 0; i < points0.size(); i++) {
        // 定义引用
        const cv::Point2f &point0 = points0[i];
        cv::Point2f &point1 = points1[i];
        uchar &matchResult = status[i];

        // 如果当前这个特征点已经追踪失败，则不再考虑
        if (matchResult == 0) {
            continue;
        }

        // 如果当前点已经追踪到了图像边界之外，则认为追踪失败不再考虑
        if (point1.x < 0 || point1.x > image1.cols - 1 ||
            point1.y < 0 || point1.y > image1.rows - 1) {
            matchResult = 0;
            continue;
        }

        // 定义最大迭代次数
        int maxEpoch = 50;
        while (maxEpoch > 0) {
            maxEpoch--;

            // 构造 G 矩阵，G = (A_T * A)_-1 * A_T
            Eigen::Matrix2d H = all_H[i];
            Eigen::Vector2d b = Eigen::Vector2d::Zero();

            // 遍历像素块内的每一个像素，计算出 H * v = b 方程中的 b
            int idx = 0;
            for (int dx = - halfBatchSize; dx <= halfBatchSize; dx++) {
                for (int dy = - halfBatchSize; dy <= halfBatchSize; dy++) {
                    double Ixi = all_Ixi[i][idx];
                    double Iyi = all_Iyi[i][idx];
                    idx++;
                    double Iti = GetPixelValue(image1, point1.x + dx, point1.y + dy) - GetPixelValue(image0, point0.x + dx, point0.y + dy);
                    b(0, 0) -= Ixi * Iti;
                    b(1, 0) -= Iyi * Iti;
                }
            }

            // 计算出光流速度，更新追踪位置
            Eigen::Vector2d v = H.inverse() * b;

            // 如果求解失败，则认为此点无法追踪
            if (std::isnan(v(0, 0)) || std::isnan(v(1, 0))) {
                matchResult = 0;
                break;
            }

            // 更新位置
            point1.x += v(0, 0);
            point1.y += v(1, 0);

            // 如果更新之后的位置超出了图像范围，则认为跟踪失败
            if (point1.x < 0 || point1.x > image1.cols || point1.y < 0 || point1.y > image1.rows) {
                matchResult = 0;
                break;
            }

            // 如果调整量非常小，则认为收敛
            if (v.norm() < convergeThreshold) {
                matchResult = 1;
                break;
            }
        }
    }
}


/* 单层正向光流追踪特征点，采用高斯牛顿方法 */
void OpticalFlowClass::TrackFeaturesSingleLevel_GN(const cv::Mat &image0,
                                                   const cv::Mat &image1,
                                                   const std::vector<cv::Point2f> &points0,
                                                   std::vector<cv::Point2f> &points1,
                                                   std::vector<uchar> &status) {
    // 如果没有设定初值，则调整初值
    if (points1.size() != points0.size() || status.size() != points0.size()) {
        points1 = points0;
        status.resize(points0.size(), 1);
    }

    // 遍历图像 0 中的每一个特征点
    for (unsigned int i = 0; i < points0.size(); i++) {
        // 定义引用
        const cv::Point2f &point0 = points0[i];
        cv::Point2f &point1 = points1[i];
        uchar &matchResult = status[i];

        // 如果当前这个特征点已经追踪失败，则不再考虑
        if (matchResult == 0) {
            continue;
        }

        // 采用高斯牛顿法，以当前 point1 的数值为初始值开始迭代优化
        int maxEpoch = 50;
        while (maxEpoch > 0) {
            maxEpoch--;
            // 如果当前点已经追踪到了图像边界之外，则认为追踪失败不再考虑
            if (point1.x < 0 || point1.x > image1.cols - 1 ||
                point1.y < 0 || point1.y > image1.rows - 1) {
                matchResult = 0;
                break;
            }

            // 构造高斯牛顿方程
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();

            // 遍历像素块内的每一个像素
            for (int dx = - halfBatchSize; dx <= halfBatchSize; dx++) {
                for (int dy = - halfBatchSize; dy <= halfBatchSize; dy++) {
                    double error = GetPixelValue(image0, point0.x + dx, point0.y + dy) -
                                   GetPixelValue(image1, point1.x + dx, point1.y + dy);
                    Eigen::Matrix<double, 1, 2> J;
                    J << GetPixelValue(image1, points1[i].x + dx + 1, points1[i].y + dy) -
                         GetPixelValue(image1, points1[i].x + dx - 1, points1[i].y + dy),
                         GetPixelValue(image1, points1[i].x + dx, points1[i].y + dy + 1) -
                         GetPixelValue(image1, points1[i].x + dx, points1[i].y + dy - 1);
                    J = - 0.5 * J;
                    b += - J.transpose() * error;
                    H += J.transpose() * J;
                }
            }

            // 求解方程并更新像素位置
            Eigen::Vector2d update = H.ldlt().solve(b);

            // 如果求解结果有问题，则认为此特征点追踪失败
            if (std::isnan(update[0]) || std::isnan(update[1])) {
                matchResult = 0;
                break;
            }

            // 更新 point1 的位置
            point1.x += update[0];
            point1.y += update[1];

            // 如果已经收敛，则追踪成功
            if (update.norm() < convergeThreshold) {
                matchResult = 1;
                break;
            }
        }
    }
}


/* 单层反向光流追踪特征点，采用高斯牛顿方法 */
void OpticalFlowClass::TrackFeaturesSingleLevel_inverse_GN(const cv::Mat &image0,
                                                           const cv::Mat &image1,
                                                           const std::vector<cv::Point2f> &points0,
                                                           std::vector<cv::Point2f> &points1,
                                                           std::vector<uchar> &status) {
    // 如果没有设定初值，则调整初值
    if (points1.size() != points0.size() || status.size() != points0.size()) {
        points1 = points0;
        status.resize(points0.size(), 1);
    }

    // 初始化图像 0 中每一个特征点对应像素块的雅可比矩阵
    std::vector<std::vector<Eigen::Vector2d>> all_J;
    all_J.resize(points0.size());
    for (unsigned int i = 0; i < points0.size(); i++) {
        // 遍历像素块内的每一个像素
        for (int dx = - halfBatchSize; dx <= halfBatchSize; dx++) {
            for (int dy = - halfBatchSize; dy <= halfBatchSize; dy++) {
                Eigen::Matrix<double, 1, 2> J;
                J << GetPixelValue(image0, points0[i].x + dx + 1, points0[i].y + dy) -
                     GetPixelValue(image0, points0[i].x + dx - 1, points0[i].y + dy),
                     GetPixelValue(image0, points0[i].x + dx, points0[i].y + dy + 1) -
                     GetPixelValue(image0, points0[i].x + dx, points0[i].y + dy - 1);
                J = - 0.5 * J;
                all_J[i].emplace_back(J);
            }
        }
    }

    // 遍历图像 0 中的每一个特征点
    for (unsigned int i = 0; i < points0.size(); i++) {
        // 定义引用
        const cv::Point2f &point0 = points0[i];
        cv::Point2f &point1 = points1[i];
        uchar &matchResult = status[i];

        // 如果当前这个特征点已经追踪失败，则不再考虑
        if (matchResult == 0) {
            continue;
        }

        // 采用高斯牛顿法，以当前 point1 的数值为初始值开始迭代优化
        int maxEpoch = 50;
        while (maxEpoch > 0) {
            maxEpoch--;
            // 如果当前点已经追踪到了图像边界之外，则认为追踪失败不再考虑
            if (point1.x < 0 || point1.x > image1.cols - 1 ||
                point1.y < 0 || point1.y > image1.rows - 1) {
                matchResult = 0;
                break;
            }

            // 构造高斯牛顿方程
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();

            // 遍历像素块内的每一个像素
            int idx = 0;
            for (int dx = - halfBatchSize; dx <= halfBatchSize; dx++) {
                for (int dy = - halfBatchSize; dy <= halfBatchSize; dy++) {
                    double error = GetPixelValue(image0, point0.x + dx, point0.y + dy) -
                                   GetPixelValue(image1, point1.x + dx, point1.y + dy);
                    Eigen::Matrix<double, 1, 2> J = all_J[i][idx];
                    idx++;
                    b += - J.transpose() * error;
                    H += J.transpose() * J;
                }
            }

            // 求解方程并更新像素位置
            Eigen::Vector2d update = H.ldlt().solve(b);

            // 如果求解结果有问题，则认为此特征点追踪失败
            if (std::isnan(update[0]) || std::isnan(update[1])) {
                matchResult = 0;
                break;
            }

            // 更新 point1 的位置
            point1.x += update[0];
            point1.y += update[1];

            // 如果已经收敛，则追踪成功
            if (update.norm() < convergeThreshold) {
                matchResult = 1;
                break;
            }
        }
    }
}


/* 多层光流金字塔追踪特征点 */
void OpticalFlowClass::TrackFeaturesMultiLevels(const cv::Mat &image0,
                                                const cv::Mat &image1,
                                                const std::vector<cv::Point2f> &points0,
                                                std::vector<cv::Point2f> &points1,
                                                std::vector<uchar> &status,
                                                uchar method) {
    // 检查输入数据是否合法
    if (image0.data == nullptr || image1.data == nullptr || points0.empty()) {
        return;
    }
    if (image0.rows != image1.rows || image0.cols != image1.cols) {
        return;
    }

    // 如果没有设定初值，则调整初值
    if (points1.size() != points0.size() || status.size() != points0.size()) {
        points1 = points0;
        status.resize(points0.size(), 1);
    }

    // 根据金字塔的层数确定比例尺
    std::vector<int> scales = {1, 2, 4, 8, 16};
    scales.resize(levelsNum);

    // 构造临时变量
    cv::Mat img0;
    cv::Mat img1;
    std::vector<cv::Point2f> pts0(points0.size());
    std::vector<cv::Point2f> pts1(points1.size());

    // 从后往前遍历每一个比例
    std::vector<int>::reverse_iterator it;
    for (it = scales.rbegin(); it != scales.rend(); it++) {
        std::cout << "scale is " << *it << std::endl;
        // 如果比例尺不是 1，需要对图像本身和 points 进行缩放
        if (*it != 1) {
            img0 = cv::Mat(image0.rows / *it, image0.cols / *it, CV_8UC1);
            img1 = cv::Mat(img0.rows, img0.cols, CV_8UC1);
            for (unsigned int u = 0; u < img0.cols; u++) {
                for (unsigned int v = 0; v < img0.rows; v++) {
                    img0.at<uchar>(v, u) = image0.at<uchar>(v * (*it), u * (*it));
                    img1.at<uchar>(v, u) = image1.at<uchar>(v * (*it), u * (*it));
                }
            }
            for (unsigned int i = 0; i < pts0.size(); i++) {
                pts0[i] = points0[i] / float(*it);
                pts1[i] = points1[i] / float(*it);
            }
        } else {
            img0 = image0;
            img1 = image1;
            pts0 = points0;
            pts1 = points1;
        }

        // 将缩放之后的图像和特征点像素坐标输入到单层光流中，根据输入参数 method 选择不同的方法
        // TODO:
        sTime = clock();
        switch (method) {
            case METHOD_LSE_FORWORD:
                TrackFeaturesSingleLevel_LSE(img0, img1, pts0, pts1, status);
                break;
            case METHOD_LSE_INVERSE:
                TrackFeaturesSingleLevel_inverse_LSE(img0, img1, pts0, pts1, status);
                break;
            case METHOD_GN_FORWORD:
                TrackFeaturesSingleLevel_GN(img0, img1, pts0, pts1, status);
                break;
            case METHOD_GN_INVERSE:
                TrackFeaturesSingleLevel_inverse_GN(img0, img1, pts0, pts1, status);
                break;
            default:
                TrackFeaturesSingleLevel_inverse_LSE(img0, img1, pts0, pts1, status);
                break;
        }

        // TODO:
        eTime = clock();
        std::cout << "Scale " << *it << " optical flow time cost " << (double)(eTime - sTime) / CLOCKS_PER_SEC << std::endl;

        // 如果比例尺不是 1，需要恢复 points1 的尺度
        if (*it != 1) {
            for (unsigned int i = 0; i < pts1.size(); i++) {
                points1[i] = pts1[i] * float(*it);
            }
        } else {
            points1 = pts1;
        }
    }
}