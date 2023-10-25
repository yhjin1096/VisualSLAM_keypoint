#include "VPS/common.hpp"
#include "VPS/node.hpp"
#include "VPS/camera.hpp"
#include "VPS/matcher.hpp"

inline cv::Affine3f trackingFeatures(const Node& prev_node, const Node& curr_node)
{
    cv::Affine3f result;

    int j = 0;
    std::vector<cv::Point3f> point_cloud;
    std::vector<cv::Point2f> point_pixel;

    Correspondence corr = Matcher::KnnMatchingORB(prev_node.left_cam.descriptor, curr_node.left_cam.descriptor);
    for(int i = 0; i < corr.match_in.size(); i++)
    {
        int q_idx = corr.match_in[i].queryIdx;
        int t_idx = corr.match_in[i].trainIdx;
        bool match = false;
        for(j = 0; j < prev_node.stereo_match.match_in.size(); j++)
        {
            if(prev_node.stereo_match.match_in[j].queryIdx == q_idx)
            {
                match = true;
                break;
            }
        }
        if(match)
        {
            point_cloud.push_back(cv::Point3f(prev_node.points_3d[j](0),prev_node.points_3d[j](1),prev_node.points_3d[j](2)));
            point_pixel.push_back(curr_node.left_cam.keypoint[t_idx].pt);
        }
    }
    cv::Mat dist_coeff = cv::Mat::zeros(5, 1, CV_64F), rvec, tvec, rot_mat;
    std::vector<int> inlier;
    cv::solvePnPRansac(point_cloud, point_pixel, prev_node.left_cam.K, dist_coeff, rvec, tvec, false, 500, 2, 0.99, inlier);
    cv::Rodrigues(rvec, rot_mat);
    std::cout << "-----------Pose Estimation-----------" << std::endl;
    std::cout << rot_mat.inv() << std::endl;
    std::cout << -rot_mat.inv()*tvec << std::endl;
    std::cout << "----------------------" << std::endl;

    result.rotation(rot_mat);
    result.translation(tvec);

    
    return result;
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

        // left_image.at<cv::Vec3b>(left_pt.y, left_pt.x)[0]
        cv::circle(left_image, left_pt, 5, cv::Scalar(0,0,255), 2);
        cv::circle(right_image, right_pt, 5, cv::Scalar(0,0,255), 2);

        Vector3d tt;
        tt << left_pt.x, left_pt.y, 1;
        tt = node.left_cam.intrinsic.inverse()*tt;
        std::cout << tt.transpose() << std::endl;
        std::cout << node.points[i].x/node.points[i].z << "," << node.points[i].y/node.points[i].z << "," << 1 << std::endl;

        Matrix3_4d transform;
        transform << 1, 0, 0, -node.base_line,
                     0, 1, 0, 0,
                     0, 0, 1, 0;

        Vector4d test_point;
        double m = 1.0;
        test_point << node.points[i].x*m, node.points[i].y*m, node.points[i].z*m, 1;
        // test_point << node.points_3d[i](0), node.points_3d[i](1), node.points_3d[i](2), 1;

        // Vector3d lp = node.left_cam.intrinsic*node.points_3d[i];
        Vector3d lp = node.left_cam.intrinsic*Vector3d(test_point(0),test_point(1),test_point(2));
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

int main(int argc, char** argv)
{
    std::vector<Node> nodes;
    uint image_index = 0;

    while(1)
    {
        Node node;
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

        node.stereo_match = Matcher::KnnMatchingORB(node.left_cam.descriptor, node.right_cam.descriptor);
        // Matcher::DrawMatching(node.left_cam, node.right_cam, node.stereo_match, "stereo_match");

        node.triangulation();
        // node.Get3DPoints();
        //projection 시켜서 확인
        ProjectionStereo(node);

        // nodes.push_back(node);
        // if(nodes.size() > 1)
        // {
        //     trackingFeatures(nodes[nodes.size()-2],nodes[nodes.size()-1]);
        //     //prev node, current node 간의 correspondence 찾고
        //     //prev node에서 3d point 계산 후
        //     //pose estimation -> 예제코드 좀 더 추가
        //     //correspondence 맞는지 확인해야함
        // }

        image_index++;
    }
    
    
    return 0;
}