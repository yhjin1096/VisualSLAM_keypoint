#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>

#include <chrono>
#include <thread>

#include <ros/ros.h>

#include <ceres/ceres.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

typedef Eigen::Matrix<double,4,4> Matrix4d;
typedef Eigen::Matrix<double,4,3> Matrix4_3d;
typedef Eigen::Matrix<double,3,3> Matrix3d;
typedef Eigen::Matrix<double,3,4> Matrix3_4d;
typedef Eigen::Matrix<double,2,2> Matrix2d;
typedef Eigen::Matrix<double,2,1> Vector2d;
typedef Eigen::Matrix<double,3,1> Vector3d;
typedef Eigen::Matrix<double,4,1> Vector4d;
typedef Eigen::Matrix<double,5,1> Vector5d;
typedef Eigen::Matrix<double,6,1> Vector6d;
typedef Eigen::Matrix<float,2,1> Vector2f;
typedef Eigen::Matrix<float,6,1> Vector6f;
typedef Eigen::Matrix<float,72,1> Vector72f;
typedef Eigen::Matrix<double,36,1> Vector36d;

typedef Eigen::Matrix<double,8,1> Vector8d;
typedef Eigen::Matrix<double,12,1> Vector12d;

typedef struct _pose_t {
  _pose_t() : R( Matrix3d::Identity() ), t( Vector3d::Zero() ) {}
  _pose_t( const _pose_t & T ) : R( T.R ), t( T.t ) {}
  _pose_t( Matrix3d R, Vector3d t ) : R(R), t(t) {}

  Matrix3d R;
  Vector3d t;
} pose_t;

class Correspondence
{
    public:
        std::vector<cv::DMatch> match_ori; //q:cam0 -> t:cam1 //모든 matching
        std::vector<cv::DMatch> match_in; //q:cam0 -> t:cam1 //matching inlier
        std::vector<int> is_in; //is_in[ori idx] -> in idx

        bool has_correspondence(int q_idx) { return (match_ori.size()!=0 && is_in[q_idx] >= 0); }; //TODO TODAY
        bool empty() { return match_ori.empty(); };
        size_t size() { return match_ori.size(); };
        cv::DMatch& operator[](int idx) { return match_ori[idx]; };
};

class Timer
{
private:
	double time1, time2;
public:
	Timer()
    {
        this->time1 = 0.0;
	    this->time2 = this->time1;
    }
	void tic(void){
        std::chrono::system_clock::time_point time = std::chrono::system_clock::now();
	    this->time1 = (double)time.time_since_epoch().count() / 1000000000.0;
    }
	void toc(void){
        std::chrono::system_clock::time_point time = std::chrono::system_clock::now();
	    this->time2 = (double)time.time_since_epoch().count() / 1000000000.0;
    }
	double get_s(void)
    {
        return this->time2 - this->time1;
    }
	double get_ms(void)
    {
        return (this->time2 - this->time1) * 1000.0;
    }
};

class GTPose
{
    public:
        std::vector<pose_t> pose_eig;
        std::vector<cv::Affine3f> pose_aff;

        void readGTPose(const std::string& path)
        {
            std::ifstream file(path);
            std::string line, word;

            if(file.is_open())
            {
                while(getline(file, line))
                {
                    int i = 0, j = 0;
                    Matrix3_4d pose;
                    std::stringstream ss(line);
                    while(getline(ss, word, ' '))
                    {
                        pose(j,i) = std::stod(word);
                        i++;
                        if(i==4)
                        {
                            i=0;
                            j++;
                        }
                    }
                    pose_t tmp_pose_eig;
                    cv::Affine3f tmp_pose_aff;
                    tmp_pose_eig.R = pose.topLeftCorner(3,3);
                    tmp_pose_eig.t = pose.topRightCorner(3,1);
                    tmp_pose_aff.rotation((cv::Mat_<float>(3,3) << pose(0,0), pose(0,1), pose(0,2),
                                                                  pose(1,0), pose(1,1), pose(1,2),
                                                                  pose(2,0), pose(2,1), pose(2,2)));
                    tmp_pose_aff.translation(cv::Vec3f(pose(0,3), pose(1,3), pose(2,3)));
                    pose_eig.push_back(tmp_pose_eig);
                    pose_aff.push_back(tmp_pose_aff);
                }
                file.close();
            }
            else
            {
                std::cout << "file not found" << std::endl;
                exit(0);
            }
        };
    private:
};

#endif