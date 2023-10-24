#ifndef __MATCHER_H__
#define __MATCHER_H__

#include "VPS/camera.hpp"
#include "VPS/common.hpp"

class Matcher
{
    public:
        static Correspondence KnnMatchingORB(const cv::Mat& query, const cv::Mat& train);
        static Correspondence BFMatchingORB(const cv::Mat& query, const cv::Mat& train);
        static void DrawMatching(const Camera& cam1, const Camera& cam2, const Correspondence& corr, const std::string& win_name);
    private:
};

#endif