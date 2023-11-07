#include "VPS/common.hpp"
#include "VPS/node.hpp"
#include "VPS/camera.hpp"
#include "VPS/matcher.hpp"

#include "pcl-1.12/pcl/io/pcd_io.h"
#include "pcl-1.12/pcl/visualization/pcl_visualizer.h"
#include "pcl-1.12/pcl/visualization/cloud_viewer.h"
#include "pcl-1.12/pcl/registration/icp.h"

inline void PoseEstimation_Essential(const Node& prev_node, Node& curr_node)
{
    cv::Affine3f result;

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    Correspondence matches = Matcher::KnnMatchingORB(prev_node.left_cam.descriptor, curr_node.left_cam.descriptor);
    for (int i = 0; i < (int) matches.size(); i++)
    {
        points1.push_back(prev_node.left_cam.keypoint[matches[i].queryIdx].pt);
        points2.push_back(curr_node.left_cam.keypoint[matches[i].trainIdx].pt);
    }
    
//   cv::Mat fundamental_matrix;
//   fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
//   std::cout << "fundamental_matrix is " << std::endl << fundamental_matrix << std::endl;

    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, prev_node.left_cam.K);
//   std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

//   cv::Mat homography_matrix;
//   homography_matrix = cv::findHomography(points1, points2, 0, 3);
//   std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;

    cv::Mat R_, t_, R, t;
    cv::recoverPose(essential_matrix, points1, points2, prev_node.left_cam.K, R_, t_);
    std::cout << "R is " << std::endl << R_.inv() << std::endl;
    std::cout << "t is " << std::endl << -R_.inv()*t_ << std::endl;
    
    R = R_.inv();
    t = -R_.inv()*t_;

    pose_t relative_pose, world_pose;
    Matrix4d transformation(Matrix4d::Identity());

    relative_pose.R << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
                       R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
                       R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    relative_pose.t << t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0);
    world_pose.R = prev_node.pose.R * relative_pose.R;
    world_pose.t = prev_node.pose.R * relative_pose.t + prev_node.pose.t;
    curr_node.pose = world_pose;
    // transformation.topLeftCorner(3,3) = curr_node.pose.R;
    // transformation.topRightCorner(3,1) = curr_node.pose.t;

    // curr_node.left_cam.projection_mat = curr_node.left_cam.intrinsic * transformation;
    // std::cout << curr_node.left_cam.projection_mat << std::endl;

    R = (cv::Mat_<float>(3,3) << world_pose.R(0,0), world_pose.R(0,1), world_pose.R(0,2),
                                        world_pose.R(1,0), world_pose.R(1,1), world_pose.R(1,2),
                                        world_pose.R(2,0), world_pose.R(2,1), world_pose.R(2,2));
    t = (cv::Mat_<float>(3,1) << world_pose.t(0,0), world_pose.t(1,0), world_pose.t(2,0));


    curr_node.R = R;
    curr_node.t = t;
    
    R.convertTo(R, CV_32F);
    result.rotation(R);
    result.translation(t);

    curr_node.pose_aff = result;
}

inline cv::Affine3f PoseEstimation_PnP(const Node& prev_node, Node& curr_node)
{
    cv::Affine3f result;

    int j = 0;
    std::vector<cv::Point3f> point_cloud;
    std::vector<cv::Point2f> point_pixel;
    std::vector<cv::Point3f> in_point_cloud;
    std::vector<cv::Point2f> in_point_pixel;

    //prev_node.left_cam.descriptor -> query, curr_node.left_cam.descriptor -> train
    Correspondence corr = Matcher::KnnMatchingORB(prev_node.left_cam.descriptor, curr_node.left_cam.descriptor);

    std::vector<cv::DMatch> tmp_corr;
    for(int i = 0; i < corr.match_in.size(); i++)
    {
        int q_idx = corr.match_in[i].queryIdx; //prev_node left_cam
        int t_idx = corr.match_in[i].trainIdx; //curr_node left_cam
        bool prev_match = false, curr_match = false;
        for(int j = 0; j < prev_node.stereo_match.match_in.size(); j++)
        {
            if(prev_node.stereo_match.match_in[j].queryIdx == q_idx)
            {
                prev_match = true;
                break;
            }
        }
        for(int j = 0; j < curr_node.stereo_match.match_in.size(); j++)
        {
            if(curr_node.stereo_match.match_in[j].queryIdx == t_idx)
            {
                curr_match = true;
                break;
            }
        }
        if(prev_match && curr_match)
        {   
            tmp_corr.push_back(corr.match_in[i]);
        }
    }

    for(int i = 0; i < tmp_corr.size(); i++)
    {
        int q_idx = tmp_corr[i].queryIdx; //prev_node left_cam
        int t_idx = tmp_corr[i].trainIdx; //curr_node left_cam
        int j = 0;
        for(j = 0; j < prev_node.stereo_match.match_in.size(); j++)
        {
            if(prev_node.stereo_match.match_in[j].queryIdx == q_idx)
                break;
        }
        if(prev_node.points_3d[j](2) > 100.0 || prev_node.points_3d[j](2) < 0)
            continue;
        // std::cout << prev_node.points_3d[j].transpose() << std::endl;
        point_cloud.push_back((cv::Point3f(prev_node.points_3d[j](0),prev_node.points_3d[j](1),prev_node.points_3d[j](2))));
        // std::cout << cv::Point3f(prev_node.points_3d[j](0),prev_node.points_3d[j](1),prev_node.points_3d[j](2)) << std::endl;
        point_pixel.push_back(curr_node.left_cam.keypoint[t_idx].pt);
        // std::cout << curr_node.left_cam.keypoint[t_idx].pt << std::endl;
    }

    //ransac으로 inlier 검출
    cv::Mat dist_coeff = cv::Mat::zeros(5, 1, CV_64F), rvec, tvec, tvec_, rot_mat, rot_mat_;
    std::vector<int> inlier;//inlier index
    
    cv::solvePnPRansac(point_cloud, point_pixel, prev_node.left_cam.K, dist_coeff, rvec, tvec, false, 500, 2, 0.99, inlier);

    //inlier 기반으로 pose estimation
    for(int i = 0; i < inlier.size(); i++)
    {
        in_point_cloud.push_back(point_cloud[inlier[i]]);
        in_point_pixel.push_back(point_pixel[inlier[i]]);
    }
    
    cv::solvePnP(in_point_cloud, in_point_pixel, prev_node.left_cam.K, dist_coeff, rvec, tvec);

    cv::Rodrigues(rvec, rot_mat);
    rot_mat_ = rot_mat.inv();
    tvec_ = -rot_mat.inv()*tvec;

    // for(int i = 0; i < point_cloud.size(); i++)
    //     std::cout << point_cloud[i].x << "," << point_cloud[i].y << "," << point_cloud[i].z << std::endl;

    std::cout << "-----------Pose Estimation-----------" << std::endl;
    std::cout << rot_mat_ << std::endl;
    std::cout << tvec_ << std::endl;
    std::cout << "----------------------" << std::endl;


    pose_t relative_pose, world_pose;
    Matrix4d transformation(Matrix4d::Identity());

    relative_pose.R << rot_mat_.at<double>(0,0), rot_mat_.at<double>(0,1), rot_mat_.at<double>(0,2),
                       rot_mat_.at<double>(1,0), rot_mat_.at<double>(1,1), rot_mat_.at<double>(1,2),
                       rot_mat_.at<double>(2,0), rot_mat_.at<double>(2,1), rot_mat_.at<double>(2,2);
    relative_pose.t << tvec_.at<double>(0,0), tvec_.at<double>(1,0), tvec_.at<double>(2,0);
    world_pose.R = prev_node.pose.R * relative_pose.R;
    world_pose.t = prev_node.pose.R * relative_pose.t + prev_node.pose.t;
    curr_node.pose = world_pose;
    // transformation.topLeftCorner(3,3) = curr_node.pose.R;
    // transformation.topRightCorner(3,1) = curr_node.pose.t;

    // curr_node.left_cam.projection_mat = curr_node.left_cam.intrinsic * transformation;
    // std::cout << curr_node.left_cam.projection_mat << std::endl;

    rot_mat_ = (cv::Mat_<float>(3,3) << world_pose.R(0,0), world_pose.R(0,1), world_pose.R(0,2),
                                        world_pose.R(1,0), world_pose.R(1,1), world_pose.R(1,2),
                                        world_pose.R(2,0), world_pose.R(2,1), world_pose.R(2,2));
    tvec_ = (cv::Mat_<float>(3,1) << world_pose.t(0,0), world_pose.t(1,0), world_pose.t(2,0));


    curr_node.R = rot_mat_;
    curr_node.t = tvec_;


    rot_mat_.convertTo(rot_mat_, CV_32F);
    result.rotation(rot_mat_);
    result.translation(tvec_);

    curr_node.pose_aff = result;
    
    return result;
}

inline void PoseEstimation_ICP(const Node& prev_node, Node& curr_node)
{
    pcl::PointCloud<pcl::PointXYZ> result_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);


    //inlier만
    bool prev_check = false, curr_check = false;
    Correspondence corr = Matcher::KnnMatchingORB(prev_node.left_cam.descriptor, curr_node.left_cam.descriptor);
    for(int i = 0; i < corr.match_in.size(); i++)
    {
        int q_idx = corr.match_in[i].queryIdx; //prev left
        int t_idx = corr.match_in[i].trainIdx; //curr left
        for(int j = 0; j < prev_node.stereo_match.match_in.size(); j++)
        {
            if(q_idx == prev_node.stereo_match.match_in[j].queryIdx)
            {
                prev_check=true;
                break;
            }
        }
        for(int j = 0; j < curr_node.stereo_match.match_in.size(); j++)
        {
            if(t_idx == curr_node.stereo_match.match_in[j].queryIdx)
            {
                curr_check=true;
                break;
            }
        }
        if(prev_check && curr_check)
        {
           cloud->points.push_back(pcl::PointXYZ(prev_node.points_3d[i](0),prev_node.points_3d[i](1),prev_node.points_3d[i](2))); 
           cloud2->points.push_back(pcl::PointXYZ(curr_node.points_3d[i](0),curr_node.points_3d[i](1),curr_node.points_3d[i](2)));
        }
        prev_check = false;
        curr_check = false;
    }
    std::cout << cloud->points.size() << std::endl;
    std::cout << cloud2->points.size() << std::endl;
    // for(int i = 0; i < prev_node.points_3d.size(); i++)
    //     cloud->points.push_back(pcl::PointXYZ(prev_node.points_3d[i](0),prev_node.points_3d[i](1),prev_node.points_3d[i](2)));
    // for(int i = 0; i < curr_node.points_3d.size(); i++)
    //     cloud2->points.push_back(pcl::PointXYZ(curr_node.points_3d[i](0),curr_node.points_3d[i](1),curr_node.points_3d[i](2)));


    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud);
    icp.setInputTarget(cloud2);
    icp.setMaximumIterations(1000);
    icp.align(result_cloud);
    std::cout << "has converged:" << icp.hasConverged() << std::endl;
    // std::cout << " score: " << icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;
    Eigen::Matrix4f icp_result = icp.getFinalTransformation();
    Matrix4d test;
    test << icp_result(0, 0), icp_result(0, 1), icp_result(0, 2), icp_result(0, 3),
            icp_result(1, 0), icp_result(1, 1), icp_result(1, 2), icp_result(1, 3),
            icp_result(2, 0), icp_result(2, 1), icp_result(2, 2), icp_result(2, 3),
            icp_result(3, 0), icp_result(3, 1), icp_result(3, 2), icp_result(3, 3);
    std::cout << test.inverse() << std::endl;
    curr_node.pose.R = test.inverse().topLeftCorner(3,3);
    curr_node.pose.t = test.inverse().topRightCorner(3,1);
}

inline void Projection(const Node& prev_node, const Node& curr_node)
{
    //visualize keypoint
    cv::Mat prev_limg = prev_node.left_cam.original_image.clone(),
            prev_rimg = prev_node.right_cam.original_image.clone(),
            curr_limg = curr_node.left_cam.original_image.clone(),
            curr_rimg = curr_node.right_cam.original_image.clone(),
            prev_image, curr_image;

    for(int i = 0; i < prev_node.stereo_match.match_in.size(); i++)
    {
        int q_idx = prev_node.stereo_match.match_in[i].queryIdx;
        int t_idx = prev_node.stereo_match.match_in[i].trainIdx;

        cv::circle(prev_limg, prev_node.left_cam.keypoint[q_idx].pt, 5, cv::Scalar(0,0,255), 2);
        cv::circle(prev_rimg, prev_node.right_cam.keypoint[t_idx].pt, 5, cv::Scalar(0,0,255), 2);
    }
    for(int i = 0; i < curr_node.stereo_match.match_in.size(); i++)
    {
        int q_idx = curr_node.stereo_match.match_in[i].queryIdx;
        int t_idx = curr_node.stereo_match.match_in[i].trainIdx;

        cv::circle(curr_limg, curr_node.left_cam.keypoint[q_idx].pt, 5, cv::Scalar(0,0,255), 2);
        cv::circle(curr_rimg, curr_node.right_cam.keypoint[t_idx].pt, 5, cv::Scalar(0,0,255), 2);
    }

    //visualize tracked features
    int prev_idx = 0, curr_idx = 0;
    std::vector<cv::Point3f> point_cloud;
    std::vector<cv::Point2f> point_pixel;
    std::vector<std::pair<int,int>> point_corr;

    //prev_node.left_cam.descriptor -> train, curr_node.left_cam.descriptor -> query
    //find correspondece between prev_node and curr_node
    Correspondence corr = Matcher::KnnMatchingORB(prev_node.left_cam.descriptor, curr_node.left_cam.descriptor);

    std::vector<cv::DMatch> tmp_corr;
    for(int i = 0; i < corr.match_in.size(); i++)
    {
        int q_idx = corr.match_in[i].queryIdx; //prev_node left_cam
        int t_idx = corr.match_in[i].trainIdx; //curr_node left_cam
        bool prev_match = false, curr_match = false;
        for(int j = 0; j < prev_node.stereo_match.match_in.size(); j++)
        {
            if(prev_node.stereo_match.match_in[j].queryIdx == q_idx)
            {
                prev_match = true;
                break;
            }
        }
        for(int j = 0; j < curr_node.stereo_match.match_in.size(); j++)
        {
            if(curr_node.stereo_match.match_in[j].queryIdx == t_idx)
            {
                curr_match = true;
                break;
            }
        }
        if(prev_match && curr_match)
        {   
            tmp_corr.push_back(corr.match_in[i]);
        }
    }

    for(int i = 0; i < tmp_corr.size(); i++)
    {
        int q_idx = tmp_corr[i].queryIdx; //prev_node left_cam
        int t_idx = tmp_corr[i].trainIdx; //curr_node left_cam
        int j = 0;
        for(j = 0; j < prev_node.stereo_match.match_in.size(); j++)
        {
            if(prev_node.stereo_match.match_in[j].queryIdx == q_idx)
                break;
        }
        if(prev_node.points_3d[j](2) > 100.0 || prev_node.points_3d[j](2) < 0)
            continue;
        // std::cout << prev_node.points_3d[j].transpose() << std::endl;
        point_cloud.push_back((cv::Point3f(prev_node.points_3d[j](0),prev_node.points_3d[j](1),prev_node.points_3d[j](2))));
        // std::cout << cv::Point3f(prev_node.points_3d[j](0),prev_node.points_3d[j](1),prev_node.points_3d[j](2)) << std::endl;
        point_pixel.push_back(curr_node.left_cam.keypoint[t_idx].pt);
        // std::cout << curr_node.left_cam.keypoint[t_idx].pt << std::endl;
    }

    //point clou를 world기준으로 변경, projection
    for(int i = 0; i < point_cloud.size(); i++)
    {
        Vector3d point(point_cloud[i].x, point_cloud[i].y, point_cloud[i].z);
        Vector3d point_ = prev_node.pose.R * point + prev_node.pose.t; // world 기준 3d point
        
        pose_t prev_pose, curr_pose; // camera 기준 world
        prev_pose.R = prev_node.pose.R.inverse();
        prev_pose.t = -prev_node.pose.R.inverse()*prev_node.pose.t;

        curr_pose.R = curr_node.pose.R.inverse();
        curr_pose.t = -curr_node.pose.R.inverse()*curr_node.pose.t;

        Vector3d prev_lp = prev_node.left_cam.intrinsic*(prev_pose.R * point_ + prev_pose.t);
        Vector3d prev_rp;
        Vector3d curr_lp = curr_node.left_cam.intrinsic*(curr_pose.R * point_ + curr_pose.t);
        Vector3d curr_rp;

        if(prev_lp(2) == 0 || curr_lp(2) == 0)
            continue;
            
        prev_lp /= prev_lp(2);
        curr_lp /= curr_lp(2);

        // std::cout << prev_lp.transpose() << std::endl;
        // std::cout << curr_lp.transpose() << std::endl;

        cv::circle(prev_limg, cv::Point(prev_lp(0), prev_lp(1)), 2, cv::Scalar(0,255,0), 2);
        // cv::circle(prev_rimg, cv::Point(prev_rp(0), prev_rp(1)), 5, cv::Scalar(0,0,255), 2);
        cv::circle(curr_limg, cv::Point(curr_lp(0), curr_lp(1)), 2, cv::Scalar(0,255,0), 2);
        // cv::circle(curr_rimg, cv::Point(curr_rp(0), curr_rp(1)), 5, cv::Scalar(0,0,255), 2);
    }

    cv::hconcat(prev_limg, prev_rimg, prev_image);
    cv::hconcat(curr_limg, curr_rimg, curr_image);
    cv::imshow("prev_image", prev_image);
    cv::imshow("curr_image", curr_image);
    char key = cv::waitKey(0);
    if(key == 27)
        exit(0);
}

inline void ProjectionStereo(const Node& node)
{
    cv::Mat left_image = node.left_cam.original_image.clone();
    cv::Mat right_image = node.right_cam.original_image.clone();

    for(int i = 0; i < node.stereo_match.match_in.size(); i++)
    {
        int q_idx = node.stereo_match.match_in[i].queryIdx;
        int t_idx = node.stereo_match.match_in[i].trainIdx;

        cv::Point2i left_pt = node.left_cam.keypoint[q_idx].pt;
        cv::Point2i right_pt = node.right_cam.keypoint[t_idx].pt;

        cv::circle(left_image, left_pt, 5, cv::Scalar(0,0,255), 2);
        cv::circle(right_image, right_pt, 5, cv::Scalar(0,0,255), 2);

        Matrix3_4d transform;
        transform << 1, 0, 0, -node.base_line,
                     0, 1, 0, 0,
                     0, 0, 1, 0;

        Vector4d test_point;
        test_point << node.points[i].x, node.points[i].y, node.points[i].z, 1;
        
        // std::cout << test_point.transpose() << std::endl;

        Vector3d lp = node.left_cam.intrinsic*test_point.head(3);
        Vector3d rp = node.right_cam.intrinsic*transform*test_point;
        lp = lp/lp(2);
        rp = rp/rp(2);
        cv::circle(left_image, cv::Point(lp(0),lp(1)), 2, cv::Scalar(0,255,0), 2);
        cv::circle(right_image, cv::Point(rp(0),rp(1)), 2, cv::Scalar(0,255,0), 2);
    }
    cv::imshow("left_image", left_image);
    cv::imshow("right_image", right_image);
    char key = cv::waitKey(0);
    if(key == 27)
        exit(0);
}

inline void VisualizeTracking(const Node& prev_node, const Node& curr_node)
{
    Correspondence corr = Matcher::KnnMatchingORB(prev_node.left_cam.descriptor, curr_node.left_cam.descriptor);

    cv::Mat prev_limg = prev_node.left_cam.original_image.clone(),
            prev_rimg = prev_node.right_cam.original_image.clone(),
            curr_limg = curr_node.left_cam.original_image.clone(),
            curr_rimg = curr_node.right_cam.original_image.clone();

    //stereo match circle
    for(int i = 0; i < prev_node.stereo_match.match_in.size(); i++)
    {
        int q_idx = prev_node.stereo_match.match_in[i].queryIdx;
        int t_idx = prev_node.stereo_match.match_in[i].trainIdx;
        // std::cout << prev_node.left_cam.keypoint.size() << "," << q_idx << prev_node.right_cam.keypoint.size() << "," << t_idx << std::endl;
        cv::circle(prev_limg, prev_node.left_cam.keypoint[q_idx].pt, 3, cv::Scalar(0,0,255), 1);
        cv::circle(prev_rimg, prev_node.right_cam.keypoint[t_idx].pt, 3, cv::Scalar(0,0,255), 1);
    }
    for(int i = 0; i < curr_node.stereo_match.match_in.size(); i++)
    {
        int q_idx = curr_node.stereo_match.match_in[i].queryIdx;
        int t_idx = curr_node.stereo_match.match_in[i].trainIdx;
        // std::cout << curr_node.left_cam.keypoint.size() << "," << q_idx << curr_node.right_cam.keypoint.size() << "," << t_idx << std::endl;
        cv::circle(curr_limg, curr_node.left_cam.keypoint[q_idx].pt, 3, cv::Scalar(0,0,255), 1);
        cv::circle(curr_rimg, curr_node.right_cam.keypoint[t_idx].pt, 3, cv::Scalar(0,0,255), 1);
    }

    //trakcing features
    std::vector<cv::DMatch> tmp_corr;
    for(int i = 0; i < corr.match_in.size(); i++)
    {
        int q_idx = corr.match_in[i].queryIdx; //prev_node left_cam
        int t_idx = corr.match_in[i].trainIdx; //curr_node left_cam
        bool prev_match = false, curr_match = false;
        for(int j = 0; j < prev_node.stereo_match.match_in.size(); j++)
        {
            if(prev_node.stereo_match.match_in[j].queryIdx == q_idx)
            {
                prev_match = true;
                break;
            }
        }
        for(int j = 0; j < curr_node.stereo_match.match_in.size(); j++)
        {
            if(curr_node.stereo_match.match_in[j].queryIdx == t_idx)
            {
                curr_match = true;
                break;
            }
        }
        if(prev_match && curr_match)
        {   
            tmp_corr.push_back(corr.match_in[i]);
        }
    }

    cv::Mat prev_image, curr_image, whole_image;
    cv::hconcat(prev_limg, prev_rimg, prev_image);
    cv::hconcat(curr_limg, curr_rimg, curr_image);

    //visualize stereo match line
    for(int i = 0; i < prev_node.stereo_match.match_in.size(); i++)
    {
        int q_idx = prev_node.stereo_match.match_in[i].queryIdx;
        int t_idx = prev_node.stereo_match.match_in[i].trainIdx;
        cv::line(prev_image, prev_node.left_cam.keypoint[q_idx].pt, prev_node.right_cam.keypoint[t_idx].pt+cv::Point2f(prev_node.left_cam.original_image.size().width,0), cv::Scalar(0,0,255), 1);
    }
    for(int i = 0; i < curr_node.stereo_match.match_in.size(); i++)
    {
        int q_idx = curr_node.stereo_match.match_in[i].queryIdx;
        int t_idx = curr_node.stereo_match.match_in[i].trainIdx;
        cv::line(curr_image, curr_node.left_cam.keypoint[q_idx].pt, curr_node.right_cam.keypoint[t_idx].pt+cv::Point2f(curr_node.left_cam.original_image.size().width,0), cv::Scalar(0,0,255), 1);
    }

    cv::vconcat(curr_image, prev_image, whole_image);

    //visualize tracking line
    for(int i = 0; i < tmp_corr.size(); i++)
    {
        int q_idx = tmp_corr[i].queryIdx; //prev_node left_cam
        int t_idx = tmp_corr[i].trainIdx; //curr_node left_cam
        cv::circle(whole_image, prev_node.left_cam.keypoint[q_idx].pt + cv::Point2f(0, curr_node.left_cam.original_image.size().height), 3, cv::Scalar(0,255,0), 2);
        cv::circle(whole_image, curr_node.left_cam.keypoint[t_idx].pt, 3, cv::Scalar(0,255,0), 2);
        cv::line(whole_image, curr_node.left_cam.keypoint[t_idx].pt, prev_node.left_cam.keypoint[q_idx].pt + cv::Point2f(0, curr_node.left_cam.original_image.size().height), cv::Scalar(0,255,0), 1);
    }

    cv::resize(whole_image, whole_image, whole_image.size()/2);
    cv::imshow("whole_image", whole_image);
    char key = cv::waitKey(0);
    if(key==27)
        exit(0);
}

int main(int argc, char** argv)
{
    std::vector<std::thread> tasks;

    std::vector<Node> nodes;
    uint image_index = 0;

    tasks.push_back(std::move(std::thread([&]()
    {
        while(1)
        {
            Node node;
            node.id = image_index;
            node.left_cam.InitCamParam();
            node.right_cam.InitCamParam();
            node.left_cam.intrinsic << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02,
                                    0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02,
                                    0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00;
            node.left_cam.projection_mat << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00,
                                            0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00,
                                            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;
            node.right_cam.intrinsic << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02,
                                        0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02,
                                        0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00;
            node.right_cam.projection_mat << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, -3.798145000000e+02,
                                            0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00,
                                            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;

            node.left_cam.LoadImage(cv::format("/home/cona/Downloads/data_odometry_gray/dataset/sequences/06/image_0/%06d.png", image_index));
            node.right_cam.LoadImage(cv::format("/home/cona/Downloads/data_odometry_gray/dataset/sequences/06/image_1/%06d.png", image_index));
            // node.left_cam.LoadImage(cv::format("/home/cona/data_odometry_gray/06/image_0/%06d.png", image_index));
            // node.right_cam.LoadImage(cv::format("/home/cona/data_odometry_gray/06/image_1/%06d.png", image_index));

            node.stereo_match = Matcher::KnnMatchingORB(node.left_cam.descriptor, node.right_cam.descriptor);
            // Matcher::DrawMatching(node.left_cam, node.right_cam, node.stereo_match, "stereo_match");

            node.triangulation();
            // node.Get3DPoints();

            // //stereo 이미지에 projection 시켜서 확인
            // ProjectionStereo(node);

            nodes.push_back(node);

            if(nodes.size() > 1)
            {
                // PoseEstimation_Essential(nodes[nodes.size()-2],nodes[nodes.size()-1]);
                PoseEstimation_PnP(nodes[nodes.size()-2],nodes[nodes.size()-1]);
                // PoseEstimation_ICP(nodes[nodes.size()-2],nodes[nodes.size()-1]);
                // VisualizeTracking(nodes[nodes.size()-2], nodes[nodes.size()-1]);
                Projection(nodes[nodes.size()-2],nodes[nodes.size()-1]);
            }

            image_index++;
        }
    })));


    GTPose gt_pose;
    gt_pose.readGTPose("/home/cona/Downloads/data_odometry_gray/data_odometry_poses/dataset/poses/06.txt");
    // gt_pose.readGTPose("/home/cona/data_odometry_gray/06/06.txt");
    
    cv::viz::Viz3d myWindow("Coordinate Frame");
    myWindow.setWindowSize(cv::Size(1280,960));
    
    tasks.push_back(std::move(std::thread([&]()
    {   
        //3d visualize
        for(int i = 0; i < gt_pose.pose_aff.size(); i++)
            myWindow.showWidget(std::to_string(i), cv::viz::WCoordinateSystem(2.0), gt_pose.pose_aff[i]);

        while(!myWindow.wasStopped())
        {
            for(int i = 0; i < nodes.size(); i++)
            {   
                myWindow.showWidget(std::to_string(nodes[i].id) + std::to_string(i), cv::viz::WCoordinateSystem(3.0), nodes[i].pose_aff);
            }
            myWindow.spinOnce(1, true);
        }
    })));
    
    for(int i=0; i<(int)tasks.size(); i++)
        tasks[i].join();
    
    return 0;
}