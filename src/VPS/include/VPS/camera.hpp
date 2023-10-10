#ifndef __CAMERA_H__
#define __CAMERA_H__

class Camera
{
    private:
        /* data */
    public:
        Camera(/* args */);
        ~Camera();

        cv::Mat original_image, gray_image;

        void InitCamParam();
        void LoadImage(const std::string& path);
};

#endif