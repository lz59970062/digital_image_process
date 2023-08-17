#ifndef __FILTER__H
#define __FILTER__H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <opencv2/core/core.hpp>

class Filter
{
public:
    cv::Mat raw_img;
    std::string filter_type;
    Filter(cv::Mat &img);
    ~Filter();
    cv::Mat meanFilter(const int &ksize);
    cv::Mat medianFilter(const int &ksize);
    cv::Mat gaussianFilter(const int &ksize, const double &sigma); 
    cv::Mat bilateralFilter(const int &ksize, const double &sigma);
    // cv::Mat laplacianFilter(const int &ksize);
    // cv::Mat sobelFilter(const int &ksize);
    cv::Mat  filter(const cv::Mat &kernel);
    cv::Mat laplace_enhence(const int ksize, bool scaled = true);
    cv::Mat sobel_enhence(bool scaled);
    cv::Mat prewitt_enhence(bool scaled);
    cv::Mat roberts_enhence(bool scaled);
    // 根据类型来生成模板
    static cv::Mat gen_gaussian_kernel(const int &ksize, const double &sigma);
    static cv::Mat gen_mean_kernel(const int &ksize);
    static cv::Mat gen_laplacian_kernel(const int &ksize);
    static cv::Mat gen_sobel_kernel(int axis);
    static cv::Mat gen_prewitt_kernel(int axis,int ksize=3);
    static cv::Mat gen_roberts_kernel(int axis,int ksize=3);

    // cv::Mat gen_bilateral_kernel(const int &ksize, const double &sigma);
    // cv::Mat gen_median_kernel(const int &ksize);
    cv::Mat gen_kernel(const std::string &type, const int &ksize, const double &sigma);
    // 其他一些滤波方法
    cv::Mat Filter::Geometric_mean_filter(const int &ksize);                       // 几何均值滤波
    cv::Mat Filter::Harmonic_mean_filter(const int &ksize);                        // 谐波均值滤波
    cv::Mat Filter::Contraharmonic_mean_filter(const int &ksize, const double &Q); // 逆谐波均值滤波
};

#endif 