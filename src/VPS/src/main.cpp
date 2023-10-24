#include "VPS/common.hpp"
#include "VPS/node.hpp"
#include "VPS/camera.hpp"
#include "VPS/matcher.hpp"

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
        Matcher::DrawMatching(node.left_cam, node.right_cam, node.stereo_match, "stereo_match");

        // node.triangulation();
        node.Get3DPoints();

        nodes.push_back(node);
        if(nodes.size() > 1)
        {
            //prev node, current node 간의 correspondence 찾고
            //prev node에서 3d point 계산 후
            //pose estimation
        }

        image_index++;
    }
    
    return 0;
}