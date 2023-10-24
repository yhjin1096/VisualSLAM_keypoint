#include "VPS/common.hpp"
#include "VPS/matcher.hpp"

Correspondence Matcher::KnnMatchingORB(const cv::Mat& query, const cv::Mat& train)
{
    if(query.empty() || train.empty())
    {
        std::cout << "descriptor not found" << std::endl;
        exit(1);
    }
    
    Correspondence output;

    const cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::IndexParams();
    indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
    indexParams->setInt("table_number", 6);
    indexParams->setInt("key_size", 12);
    indexParams->setInt("multi_probe_level", 1);

    const cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams();
    searchParams->setInt("checks", 20);

    std::vector<std::vector<cv::DMatch>> knn_match;
    cv::FlannBasedMatcher matcher(indexParams, searchParams);
    matcher.knnMatch(query, train, knn_match, 2);

    output.match_ori.resize(knn_match.size());
    output.is_in.resize(knn_match.size());
    output.match_in.reserve(knn_match.size());

    for(size_t i = 0; i < knn_match.size(); ++i)
    {
        const std::vector<cv::DMatch>& match = knn_match[i];
        
        if(match.empty())
            continue;

        const cv::DMatch& m1 = match[0];
        const cv::DMatch& m2 = match[1];
        const double& d1 = m1.distance;
        const double& d2 = m2.distance;

        output.match_ori[i] = m1;
        output.is_in[i] = -1;

        // if(d1 / d2 < 0.6 && d1 < 0.5)
        if(d1 < d2 * 0.65)
        {
            output.is_in[i] = (int)output.match_in.size();
            output.match_in.push_back(m1);
        }
    }

    return output;
}
Correspondence Matcher::BFMatchingORB(const cv::Mat& query, const cv::Mat& train)
{
    Correspondence output;
    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    if(matcher != nullptr){
        std::vector<cv::DMatch> matches;
        matcher->match(query, train, matches);
        output.match_ori = matches;

        std::sort(matches.begin(), matches.end()); //std::pair 만들고 정렬
        std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + 200);
        output.match_in = good_matches;
    }

    return output;
}

void Matcher::DrawMatching(const Camera& cam1, const Camera& cam2, const Correspondence& corr, const std::string& win_name)
{
    int q_idx, t_idx;
    std::vector<cv::KeyPoint> kp_in_1, kp_in_2;
    cv::Mat cam1_image = cam1.original_image.clone(), cam2_image = cam2.original_image.clone();

    for(int i = 0; i < corr.match_in.size(); i++)
    {
        q_idx = corr.match_in[i].queryIdx;
        t_idx = corr.match_in[i].trainIdx;

        // kp_in_1.push_back(cam1.keypoint[q_idx]);
        // kp_in_2.push_back(cam2.keypoint[t_idx]);
        // cv::putText(cam1_image,std::to_string(q_idx),kp_in_1[kp_in_1.size()-1].pt,1,1,cv::Scalar(0,0,255),1);
        // cv::putText(cam2_image,std::to_string(t_idx),kp_in_2[kp_in_2.size()-1].pt,1,1,cv::Scalar(0,0,255),1);
    }

    cv::Mat img_matches;
    cv::drawMatches( cam1_image, cam1.keypoint, cam2_image, cam2.keypoint, corr.match_in, img_matches, cv::Scalar::all(-1),
    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    // cv::resize(img_matches, img_matches, img_matches.size()/2);

    cv::imshow(win_name, img_matches);
    char key = cv::waitKey(0);
    if(key == 27)
        exit(0);
}