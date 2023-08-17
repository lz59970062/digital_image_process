#ifndef __IMG_TRANS_H__
#define __IMG_TRANS_H__

#include <opencv2/opencv.hpp>

// imagery enhancement

class Enhancement
{
public:
	Enhancement(const cv::Mat &img);
	~Enhancement();
	cv::Mat linearTransform(const double &a, const double &b);
	cv::Mat piecewiseLinearTransform(const  std::vector<double> &a, const  std::vector<double> &b);
	cv::Mat gammaTransform(const double &gamma);
	cv::Mat gammaTransform_f(const double &gamma);
	cv::Mat logTransform(double  c);
	cv::Mat logTransform_f(double  c);
	cv::Mat result_img;
private:
	cv::Mat raw_image;
};

#endif
