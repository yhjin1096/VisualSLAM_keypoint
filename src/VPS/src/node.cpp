#include "VPS/common.hpp"
#include "VPS/node.hpp"

void Node::triangulation()
{
    cv::Mat T1 = (cv::Mat_<float>(3, 4) << this->left_cam.projection_mat(0, 0), this->left_cam.projection_mat(0, 1), this->left_cam.projection_mat(0, 2), this->left_cam.projection_mat(0,3),
                                           this->left_cam.projection_mat(1, 0), this->left_cam.projection_mat(1, 1), this->left_cam.projection_mat(1, 2), this->left_cam.projection_mat(1,3),
                                           this->left_cam.projection_mat(2, 0), this->left_cam.projection_mat(2, 1), this->left_cam.projection_mat(2, 2), this->left_cam.projection_mat(2,3));
                                           
    cv::Mat T2 = (cv::Mat_<float>(3, 4) << this->right_cam.projection_mat(0, 0), this->right_cam.projection_mat(0, 1), this->right_cam.projection_mat(0, 2), this->right_cam.projection_mat(0,3),
                                           this->right_cam.projection_mat(1, 0), this->right_cam.projection_mat(1, 1), this->right_cam.projection_mat(1, 2), this->right_cam.projection_mat(1,3),
                                           this->right_cam.projection_mat(2, 0), this->right_cam.projection_mat(2, 1), this->right_cam.projection_mat(2, 2), this->right_cam.projection_mat(2,3));

    std::vector<cv::Point2f> pts_1, pts_2;
    
    for (cv::DMatch m : this->stereo_match.match_in)
    {
        //pixel coordinate
        pts_1.push_back(this->left_cam.keypoint[m.queryIdx].pt);
        pts_2.push_back(this->right_cam.keypoint[m.trainIdx].pt);
    }

    // computed triangulated points returned in world coordinate system (that is w.r.t cam1 since T1 = eye )
    cv::Mat pts_4d; 
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // Convert to non-homogeneous coordinates
    for (int i = 0; i < pts_4d.cols; i++)
    {
        cv::Mat x = pts_4d.col(i);

        if(x.at<float>(3, 0) != 0)
            x /= x.at<float>(3, 0); // Normalized

        cv::Point3d p(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0));
        this->points.push_back(p);
        this->points_3d.push_back(Vector3d(p.x, p.y, p.z));
        // std::cout << points[points.size()-1] << std::endl;
        // std::cout << points_3d[points_3d.size()-1].transpose() << std::endl;
    }
}

void Node::Get3DPoints()
{
    if(this->stereo_match.match_in.empty())
    {
        std::cout << "stereo match is empty" << std::endl;
        //TODO -> keyframe 건너뛰기?
    }
    
    int match_in_size = stereo_match.match_in.size();

    for(int i = 0; i < match_in_size; i++)
    {
        int q_idx = stereo_match.match_in[i].queryIdx;
        int t_idx = stereo_match.match_in[i].trainIdx;
        
        // Vector3d left_sp = left_cam.sphere_points[left_cam.keypoint[q_idx].pt.y][left_cam.keypoint[q_idx].pt.x];
        // Vector3d right_sp = right_cam.sphere_points[right_cam.keypoint[t_idx].pt.y][right_cam.keypoint[t_idx].pt.x];
        Vector3d left_sp, right_sp;
        left_sp << this->left_cam.keypoint[q_idx].pt.x,
                   this->left_cam.keypoint[q_idx].pt.y,
                   1.0;
        right_sp << this->right_cam.keypoint[t_idx].pt.x,
                    this->right_cam.keypoint[t_idx].pt.y,
                    1.0;
        left_sp = this->left_cam.intrinsic.inverse() * left_sp;
        right_sp = this->left_cam.intrinsic.inverse() * right_sp;
        
        //non-linear triangulation
        Matrix2d A;
        Vector3d base_line_;
        Vector2d b, result;

        base_line_ << -base_line, 0., 0.;

        // Vector3d test(0.1,0.1,1.5);
        // left_sp = test.normalized();
        // right_sp = (test - Vector3d(0.12,0,0)).normalized();

        // std::cout << left_sp.transpose() << "," << right_sp.transpose() << std::endl;

        b << -base_line_.transpose()*left_sp, -base_line_.transpose()*right_sp;

        A << left_sp.transpose()*left_sp, -right_sp.transpose()*left_sp,
             left_sp.transpose()*right_sp, -right_sp.transpose()*right_sp;
        
        result = A.inverse()*b;
        
        Vector3d F, G;
        F = result(0) * left_sp;
        G = base_line_ + result(1) * right_sp;

        // std::cout << ((F+G)/2.0 + Vector3d(base_line,0,0)).transpose() << std::endl;
        
        this->points_3d.push_back((F+G)/2.0 + Vector3d(0.12,0,0));
        // this->points_3d_map.insert(std::make_pair(q_idx, this->points_3d[points_3d.size()-1]));
        // this->points_3d_color.push_back(left_cam.image.at<cv::Vec3b>(left_cam.keypoint[q_idx].pt.y, left_cam.keypoint[q_idx].pt.x));
    }
}