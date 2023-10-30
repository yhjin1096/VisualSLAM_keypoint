#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "VPS/common.hpp"

class Camera
{
    private:
        /* data */
    public:
        // Camera(/* args */);
        // ~Camera();

        Matrix3d intrinsic;
        Matrix3_4d projection_mat;
        cv::Mat K;

        cv::Mat original_image, gray_image;
        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;

        void InitCamParam();
        void LoadImage(const std::string& path);
        void ExtractORB();
};

#endif