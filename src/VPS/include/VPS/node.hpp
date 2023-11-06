#ifndef __NODE_H__
#define __NODE_H__

#include "VPS/common.hpp"
#include "VPS/camera.hpp"

class Node
{
    private:
        /* data */
    public:
        // Node(/* args */);
        // ~Node();
        uint id;
        
        Camera left_cam, right_cam;
        double base_line = 0.537150653;
        cv::Mat R = cv::Mat::eye(3, 3, CV_32F), t = cv::Mat::zeros(3, 1, CV_32F);
        pose_t pose; //world 기준 node 위치
        cv::Affine3f pose_aff;

        Correspondence stereo_match;
        std::vector<cv::Point3d> points; //3d point
        std::vector<Vector3d> points_3d; //3d point

        void triangulation();
        void Get3DPoints();
};

inline cv::Point2f pixel2cam(const cv::Point2d &p, const cv::Mat &K)
{
    return cv::Point2f(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

#endif