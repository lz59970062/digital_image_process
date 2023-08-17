#include "filter.h"

Filter::Filter(cv::Mat &img)
{
    if (img.empty())
    {
        std::cerr << "Image is empty!" << std::endl;
    }
    if (img.channels() == 3)
    {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    this->raw_img = img.clone();
}
Filter::~Filter()
{
    std::cout << "Filter class destructed" << std::endl;
}
cv::Mat Filter::meanFilter(const int &ksize)
{
    this->filter_type = "mean";
    cv::Mat result_img = this->raw_img.clone();
    cv::blur(this->raw_img, result_img, cv::Size(ksize, ksize));
    return result_img;
}
cv::Mat Filter::medianFilter(const int &ksize)
{
    this->filter_type = "median";
    cv::Mat result_img = this->raw_img.clone();
    cv::medianBlur(this->raw_img, result_img, ksize);
    return result_img;
}
cv::Mat Filter::gaussianFilter(const int &ksize, const double &sigma)
{
    this->filter_type = "gaussian";
    cv::Mat result_img = this->raw_img.clone();
    cv::GaussianBlur(this->raw_img, result_img, cv::Size(ksize, ksize), sigma);
    return result_img;
}
cv::Mat Filter::bilateralFilter(const int &ksize, const double &sigma)
{
    this->filter_type = "bilateral";
    cv::Mat result_img = this->raw_img.clone();
    cv::bilateralFilter(this->raw_img, result_img, ksize, sigma, sigma);
    return result_img;
}
// cv::Mat Filter::laplacianFilter(const int &ksize)
// {
//     this->filter_type = "laplacian";
//     cv::Mat result_img;
//     cv::Laplacian(this->raw_img, result_img, CV_16S, ksize);
//     cv::convertScaleAbs(result_img, result_img);
//     return result_img;
// }
// cv::Mat Filter::sobelFilter(const int &ksize)
// {
//     this->filter_type = "sobel";
//     cv::Mat result_img = this->raw_img.clone();
//     cv::Sobel(this->raw_img, result_img, CV_16S, 1, 1, ksize);
//     cv::convertScaleAbs(result_img, result_img);

//     return result_img;
// }

// 根据模板来进行滤波
cv::Mat Filter::filter(const cv::Mat &kernel)
{
    this->filter_type = "mask filter";
    cv::Mat result_img = this->raw_img.clone();
    cv::filter2D(this->raw_img, result_img, -1, kernel);
    return result_img;
}

cv::Mat Filter::gen_gaussian_kernel(const int &ksize, const double &sigma)
{
    // cv::Mat kernel=cv::getGaussianKernel(ksize, sigma);
    if (ksize % 2 == 0)
    {
        std::cerr << "ksize must be odd!" << std::endl;
        return cv::Mat();
    }
    if (sigma <= 0)
    {
        std::cerr << "sigma must be positive!" << std::endl;
        return cv::Mat();
    }
    cv::Mat kernel = cv::Mat(ksize, ksize, CV_64FC1);
    int center = ksize / 2;
    double sum = 0;
    for (int i = 0; i < ksize; i++)
    {
        for (int j = 0; j < ksize; j++)
        {
            kernel.at<double>(i, j) = exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (sigma * sigma));
            sum += kernel.at<double>(i, j);
        }
    }
    kernel /= sum;
    return kernel;
}
cv::Mat Filter::gen_mean_kernel(const int &ksize)
{
    if (ksize % 2 == 0)
    {
        std::cerr << "ksize must be odd!" << std::endl;
        return cv::Mat();
    }
    cv::Mat kernel = cv::Mat(ksize, ksize, CV_64FC1, cv::Scalar(1.0 / (ksize * ksize)));
    return kernel;
}

cv::Mat Filter::gen_laplacian_kernel(const int &ksize)
{
    if (ksize % 2 == 0)
    {
        std::cerr << "ksize must be odd!" << std::endl;
        return cv::Mat();
    }
    cv::Mat kernel;
    if (ksize == 3)
    {
        kernel = (cv::Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
        // cv::Mat kernel=(cv::Mat_<double>(3,3)<<1,1,1,1,-8,1,1,1,1);
        return kernel;
    }
    if (ksize == 5)
    {
        kernel = (cv::Mat_<double>(5, 5) << 0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0);
        return kernel;
    }
}

cv::Mat Filter::gen_sobel_kernel(int axis)
{
    cv::Mat kernel;
    if (axis == 0)
    {
        kernel = (cv::Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    }
    if (axis == 1)
    {
        kernel = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    }
    return kernel;
}

// //其他一些滤波方法
cv::Mat Filter::Geometric_mean_filter(const int &ksize)
{
    this->filter_type = "geometric";
    cv::Mat result_img = this->raw_img.clone();
    if (result_img.channels() == 3)
    {
        cv::cvtColor(result_img, result_img, cv::COLOR_BGR2GRAY);
    }

    int center = ksize / 2;
    double pro = 1;
    for (int i = ksize / 2; i < result_img.rows - ksize / 2; i++)
    {
        for (int j = ksize / 2; j < result_img.cols - ksize / 2; j++)
        {
            for (int m = 0; m < ksize; m++)
            {
                for (int n = 0; n < ksize; n++)
                {
                    pro *= this->raw_img.at<uchar>(i + m - center, j + n - center);
                }
            }
            result_img.at<uchar>(i, j) = cv::saturate_cast<uchar>(pow(pro, 1.0 / (ksize * ksize)));
            pro = 1;
        }
    }
    return result_img;

} // 几何均值滤波


cv::Mat Filter::Harmonic_mean_filter(const int &ksize)
{
    this->filter_type = "harmonic_mean";
    cv::Mat result_img(this->raw_img.size(), this->raw_img.type());
    int center = ksize / 2;
    double pro = 0;
    for (int i = center; i < result_img.rows - center; i++)
    {
        for (int j = center; j < result_img.cols - center; j++)
        {
            for (int m = 0; m < ksize; m++)
            {
                for (int n = 0; n < ksize; n++)
                {
                    pro += 1.0 / this->raw_img.at<uchar>(i + m - center, j + n - center);
                }
            }
            result_img.at<uchar>(i, j) = cv::saturate_cast<uchar>(ksize * ksize / pro);
            pro = 0;
        }
    }
    return result_img;
} // 谐波均值滤波

cv::Mat Filter::Contraharmonic_mean_filter(const int &ksize, const double &Q)
{
    this->filter_type = "contraharmonic_mean";
    cv::Mat result_img(this->raw_img.size(), this->raw_img.type());
    int center = ksize / 2;
    double pro1 = 0, pro2 = 0;
    for (int i = center; i < result_img.rows - center; i++)
    {
        for (int j = center; j < result_img.cols - center; j++)
        {
            for (int m = 0; m < ksize; m++)
            {
                for (int n = 0; n < ksize; n++)
                {
                    pro1 += pow(this->raw_img.at<uchar>(i + m - center, j + n - center), Q + 1);
                    pro2 += pow(this->raw_img.at<uchar>(i + m - center, j + n - center), Q);
                }
            }
            result_img.at<uchar>(i, j) = cv::saturate_cast<uchar>(pro1 / pro2);
            pro1 = 0;
            pro2 = 0;
        }
    }
    return result_img;
} // 逆谐波均值滤波
