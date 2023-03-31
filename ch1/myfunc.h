#ifndef __MY_FUNC__
#define __MY_FUNC__
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
using namespace cv;
void slatNosise(cv::Mat &img,double f);
cv::Mat img_transf(Mat &img, Mat &T, int methed);
Mat Transform(double theta,double sw,double sh,double mx,double my );
#endif 