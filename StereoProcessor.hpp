#ifndef STEREO_PROCESSOR_HPP
#define STEREO_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <string>

class StereoProcessor {
public:
    double focal_px;
    double baseline_mm;
    double scaleFactor;
    cv::Size imgSize;
    cv::Mat qMatrix;

    StereoProcessor(const std::string& calibration_file, const std::string& scale_file);

    void setSGBMParameters(int numDispMult, int blockSz);
    void computeDisparity(const cv::Mat& rectL, const cv::Mat& rectR, int claheTrack, cv::Mat& dispFilt, cv::Mat& dispF);
    cv::Mat getConfidenceMap() const;
    void rectifyImages(const cv::Mat& imgLeft, const cv::Mat& imgRight, cv::Mat& rectL, cv::Mat& rectR);

private:
    cv::cuda::GpuMat d_mapL1, d_mapL2, d_mapR1, d_mapR2;
    cv::Ptr<cv::StereoSGBM> sgbm;
    cv::Ptr<cv::StereoMatcher> rightMatcher;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls;
    cv::Mat lastConfidenceMap;

    int last_nd = -1;
    int last_bs = -1;
};

#endif // STEREO_PROCESSOR_HPP
