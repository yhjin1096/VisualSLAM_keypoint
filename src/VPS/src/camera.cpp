#include "VPS/common.hpp"
#include "VPS/camera.hpp"

void Camera::InitCamParam()
{
    this->K = (cv::Mat_<double>(3, 3) << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02,
                                           0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02,
                                           0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
}

void Camera::LoadImage(const std::string& path)
{
    cv::Mat image = cv::imread(path);
    if(image.empty())
    {
        std::cout << "image not found" << std::endl;
        exit(1);
    }
    this->original_image = image.clone();
    cv::cvtColor(image, this->gray_image, CV_8UC1);
    ExtractORB();
}

void Camera::ExtractORB()
{
    cv::Ptr<cv::Feature2D> feature2d = cv::ORB::create();
    // descriptor = cv::Mat(500, 32, CV_32FC1, cv::Scalar(0));
    feature2d->detectAndCompute(this->gray_image, cv::Mat(), this->keypoint, this->descriptor);//this->gray_image

    // cv::Mat test = this->gray_image.clone();
    // for(int i = 0; i < keypoint.size(); i++)
    // {
    //     cv::circle(test, keypoint[i].pt, 3, cv::Scalar(0,0,255), 2);
    // }
    // cv::imshow("test",test);
    // cv::waitKey(0);
}