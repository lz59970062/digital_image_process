 
#include "filter.h" 
#include <algorithm>
#include <iostream>
// 1、使用前面建立的工程，参照示例 1，添加四个函数，分别实现图像的线性变换，分段线性变换，对数变换及幂变换。图像及变换系数从参数中传入。在主函数中编写代码，测试上述函数，调整不同的参数，观察图像处理的效果。给出变换的代码及测试的结果。加入代码绘制上述图像处理前后的直方图分布，并进行比较。 
// 2、 编写主函数，从磁盘中调入图像，对图像进行直方图均衡化处理，并比较处理前后图像的直方图。报告中给出直方图均衡化函数的流程图，对上述要求的实现的过程及对结果进行分析。
// 3、 打开一幅图像，对图像进行各种不同模板大小的邻域平均滤波。观察图像的变化情况并比较变化前后图像直方图的差异；编写排序函数（或使用教师给的排序函数），利用排序函数对图像进行中值滤波处理，并比较不同大小模板的效果；打开一幅灰度图像，对图像进行交叉微分算子、梯度法、及 sobel 算子的一阶锐化处理，比较图像变化前后的直方图；打开一幅图像并进行二阶拉普拉斯算子的锐化处理，并分析算子的作用。 
// 4、 建立工程，配置好环境，使用给定的快速二维傅立叶变换函数或 OpenCV 函数库的 离散傅立叶变换函数实现对给定图像的傅立叶变换，观察傅立叶变换后频谱的分布情况，并分析结果；对一幅图像进行傅立叶变换，分别设置虚部为 0 和模为常数，重构图像并观察图像，分析实验结果；读入图像，使用理想低通滤波器和布特沃斯低通滤波器对图像进行频率域的低通滤波。调整滤波器半径，观察不同半径的滤波器实现的效果，并分析原因；读入图像，对图像进行频率域的高通滤波（任意一种）。 整滤波器半径，观察不同半径的滤波器实现的效果，并分析原因；读入图像，并对图像进行傅立叶变换，请选择一对和单个频率点，其他设为零，然后进行傅立叶反变换，观察现象，并分析结果。

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
    if (result_img.type() == CV_64FC1 || result_img.type() == CV_32FC1)
    {
        int k_2 = ksize / 2;
        std::vector<double> arr;
        for (int i = 0; i < result_img.rows; i++)
        {
            for (int j = 0; j < result_img.cols; j++)
            {
                arr.clear();
                for (int m = 0; m < ksize; m++)
                {
                    for (int n = 0; n < ksize; n++)
                    {
                        int pos1 = i + m - k_2;
                        int pos2 = j + n - k_2;
                        if (pos1 >= 0 && pos1 < raw_img.rows && pos2 >= 0 && pos2 < raw_img.cols)
                        {
                            arr.push_back(raw_img.at<double>(pos1, pos2));
                        }
                    }
                }
                std::sort(arr.begin(), arr.end());
                result_img.at<double>(i, j) = arr[arr.size() / 2];
            }
        }
    }
    else
    {
        int k_2 = ksize / 2;
        std::vector<uchar> arr;
        for (int i = 0; i < result_img.rows; i++)
        {
            for (int j = 0; j < result_img.cols; j++)
            {
                arr.clear();
                for (int m = 0; m < ksize; m++)
                {
                    for (int n = 0; n < ksize; n++)
                    {
                        int pos1 = i + m - k_2;
                        int pos2 = j + n - k_2;
                        if (pos1 >= 0 && pos1 < raw_img.rows && pos2 >= 0 && pos2 < raw_img.cols)
                        {
                            arr.push_back(raw_img.at<uchar>(pos1, pos2));
                        }
                    }
                }
                std::sort(arr.begin(), arr.end());
                result_img.at<uchar>(i, j) = arr[arr.size() / 2];
            }
        }
    }
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
cv::Mat Filter::laplace_enhence(const int ksize, bool scaled)
{
    cv::Mat kernel = gen_laplacian_kernel(ksize);
    cv::Mat edge;
    if (scaled == true)
    {
        cv::filter2D(this->raw_img, edge, -1, kernel);
        return this->raw_img + edge;
    }
    else
    {
        this->raw_img.convertTo(raw_img, CV_64F);
        raw_img /= 255;
        cv::filter2D(this->raw_img, edge, -1, kernel);
        return this->raw_img + edge;
    }
}
cv::Mat Filter::sobel_enhence(bool scaled)
{
    cv::Mat kernel1 = gen_sobel_kernel(1);
    cv::Mat kernel2 = gen_sobel_kernel(0);
    cv::Mat edge1, edge2;
    if (scaled)
    {
        cv::filter2D(this->raw_img, edge1, -1, kernel1);
        cv::filter2D(this->raw_img, edge2, -1, kernel2);
        return this->raw_img + edge1 + edge2;
    }
    else
    {
        this->raw_img.convertTo(raw_img, CV_64F);
        raw_img /= 255;
        cv::filter2D(this->raw_img, edge1, -1, kernel1);
        cv::filter2D(this->raw_img, edge2, -1, kernel2);
        return this->raw_img + edge1 + edge2;
    }
}
cv::Mat Filter::prewitt_enhence(bool scaled)
{
    cv::Mat kernel1 = gen_prewitt_kernel(1);
    cv::Mat kernel2 = gen_prewitt_kernel(0);
    cv::Mat edge1, edge2;
    if (scaled)
    {
        cv::filter2D(this->raw_img, edge1, -1, kernel1);
        cv::filter2D(this->raw_img, edge2, -1, kernel2);
        return this->raw_img + edge1 + edge2;
    }
    else
    {
        this->raw_img.convertTo(raw_img, CV_64F);
        raw_img /= 255;
        cv::filter2D(this->raw_img, edge1, -1, kernel1);
        cv::filter2D(this->raw_img, edge2, -1, kernel2);
        return this->raw_img + edge1 + edge2;
    }
}
cv::Mat Filter::roberts_enhence(bool scaled)
{
    cv::Mat kernel1 = gen_roberts_kernel(1);
    cv::Mat kernel2 = gen_roberts_kernel(0);
    cv::Mat edge1, edge2;  
    if (scaled)
    {
        cv::filter2D(this->raw_img, edge1, -1, kernel1);
        cv::filter2D(this->raw_img, edge2, -1, kernel2);
        return this->raw_img + edge1 + edge2;
    }
    else
    {
        this->raw_img.convertTo(raw_img, CV_64F);
        raw_img /= 255;
        cv::filter2D(this->raw_img, edge1, -1, kernel1);
        cv::filter2D(this->raw_img, edge2, -1, kernel2);
        return this->raw_img + edge1 + edge2;
    }
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

cv::Mat Filter::gen_prewitt_kernel(int axis, int ksize)
{
    if (ksize == 3)
    {
        if (axis == 0)
        {
            cv::Mat kernel = (cv::Mat_<double>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
            return kernel;
        }
        if (axis == 1)
        {
            cv::Mat kernel = (cv::Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
            return kernel;
        }
    }
}
cv::Mat Filter::gen_roberts_kernel(int axis, int ksize)
{
    if (axis == 0)
    {
        cv::Mat kernel = (cv::Mat_<double>(2, 2) << 1, 0, 0, -1);
        return kernel;
    }
    if (axis == 1)
    {
        cv::Mat kernel = (cv::Mat_<double>(2, 2) << 0, 1, -1, 0);
        return kernel;
    }
}

