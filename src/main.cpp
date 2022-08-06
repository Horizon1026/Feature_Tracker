#include <opticalFlow.h>
#include <FASTFeatureDetector.h>
#include <ctime>
// 定义计时用变量
clock_t startTime, endTime;

// 定义两张彩色图像的存放路径
std::string image0_filepath = "../examples/image0.png";
std::string image1_filepath = "../examples/image1.png";

int main() {
    std::cout << "Optical Flow Lib Test" << std::endl;
    OpticalFlowClass OpticalFlow(4, 10);
    FASTFeatureDetectorClass FASTFeatureDetector(0.2, 3, 9, 15, 10);

    // 加载两幅图像（以 U8C1 方式），并检查是否加载成功，如果不成功则终止程序
    cv::Mat image0 = cv::imread(image0_filepath, 0);
    cv::Mat image1 = cv::imread(image1_filepath, 0);
    assert(image0.data != nullptr && image1.data != nullptr);
    std::cout << "image rows " << image1.rows << ", cols " << image1.cols << std::endl;

    // 在图像 0 中检测 FAST 角点
    std::vector<cv::Point2f> points0 = FASTFeatureDetector.DetectGoodSparseFeatures(image0);
    // 在原图上画上特征点
    cv::Mat showImage0 = cv::Mat(image0.rows, image0.cols, CV_8UC3);
    cv::cvtColor(image0, showImage0, CV_GRAY2BGR);
    for (auto &point : points0) {
        cv::circle(showImage0, point, 3, cv::Scalar(0, 0, 255), 1);
    }
    cv::imshow("image0 with FAST features", showImage0);

    /* --------------------------------------------------------------------------------------------- */
    // 在图像 1 中追踪 FAST 角点
    {
        std::vector<cv::Point2f> points1;
        std::vector<uchar> status;
        startTime = clock();
        OpticalFlow.TrackFeaturesMultiLevels(image0, image1, points0, points1, status, OpticalFlowClass::METHOD_LSE_INVERSE);
        endTime = clock();
        // 在原图上画上特征点
        cv::Mat showImage1 = cv::Mat(image1.rows, image1.cols, CV_8UC3);
        cv::cvtColor(image1, showImage1, CV_GRAY2BGR);
        int cnt = 0;
        for (unsigned int i = 0; i < points0.size(); i++) {
            if (status[i] != 1) {
                continue;
            }
            cnt++;
            cv::circle(showImage1, points1[i], 3, cv::Scalar(0, 0, 255), 1);
            cv::line(showImage1, points1[i], points0[i], cv::Scalar(0, 255, 0), 1);
        }
        cv::imshow("My code image1 with FAST features Tracked", showImage1);
        // 打印出追踪结果
        std::cout << "My code track nums " << cnt << "/" << status.size() << std::endl;
        std::cout << "Time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
        std::cout << std::endl;
    }

    /* --------------------------------------------------------------------------------------------- */
    {
        std::vector<cv::Point2f> points1;
        std::vector<uchar> status;
        std::vector<float> error;
        startTime = clock();
        cv::calcOpticalFlowPyrLK(image0, image1, points0, points1, status, error, cv::Size(21, 21), 3);
        endTime = clock();
        // 在原图上画上特征点
        cv::Mat showImage1 = cv::Mat(image1.rows, image1.cols, CV_8UC3);
        cv::cvtColor(image1, showImage1, CV_GRAY2BGR);
        int cnt = 0;
        for (unsigned int i = 0; i < points0.size(); i++) {
            if (status[i] != 1) {
                continue;
            }
            cnt++;
            cv::circle(showImage1, points1[i], 3, cv::Scalar(0, 0, 255), 1);
            cv::line(showImage1, points1[i], points0[i], cv::Scalar(0, 255, 0), 1);
        }
        cv::imshow("OpenCV image1 with FAST features Tracked", showImage1);
        // 打印出追踪结果
        std::cout << "OpenCV track nums " << cnt << "/" << status.size() << std::endl;
        std::cout << "Time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
    }

    cv::waitKey();
    return 0;
}