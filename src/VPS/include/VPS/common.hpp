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


#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"

typedef Eigen::Matrix<double,4,4> Matrix4d;
typedef Eigen::Matrix<double,4,3> Matrix4_3d;
typedef Eigen::Matrix<double,3,3> Matrix3d;
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

#endif