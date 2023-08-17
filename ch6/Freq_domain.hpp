#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <opencv2/core/core.hpp>
#include <complex>
#define PI 3.14159265358979323846
double sinc(double x)
{
    if (x == 0.0)
    {
        return 1.0;
    }
    else
    {
        return sin(x) / x;
    }
}
class Freq_domain
{
private:
    /* data */
public:
    int opt_w, opt_h;
    cv::Size opt_size;
    int raw_w, raw_h;
    cv::Mat dft_img;
    cv::Mat raw_img;
    cv::Mat dft_img_to_show;
    cv::Mat dft_img_phase_show;
    cv::Mat dft_img_real, dft_img_imag;

    Freq_domain(/* args */)
    {
        opt_w = 0;
        opt_h = 0;
    }
    Freq_domain(cv::Mat &img)
    {

        Mat planes[] = {Mat_<double>(img), Mat::zeros(img.size(), CV_64F)};
        raw_img = img.clone();
        this->dft_img_to_show = dft(img);
        cv::split(this->dft_img, planes);
        fftshift(planes[0]);
        fftshift(planes[1]);
        this->dft_img_real = planes[0];
        this->dft_img_imag = planes[1];
        cv::phase(dft_img_imag, dft_img_real, dft_img_phase_show);
        cv::normalize(dft_img_phase_show, dft_img_phase_show, 0, 1, cv::NORM_MINMAX);
    }
    ~Freq_domain() {}
    inline static void fftshift(cv::Mat &magImage)
    {
        magImage = magImage(Rect(0, 0, magImage.cols & -2, magImage.rows & -2));
        int cx = magImage.cols / 2;
        int cy = magImage.rows / 2;
        Mat q0(magImage, Rect(0, 0, cx, cy));
        Mat q1(magImage, Rect(cx, 0, cx, cy));
        Mat q2(magImage, Rect(0, cy, cx, cy));
        Mat q3(magImage, Rect(cx, cy, cx, cy));
        // 交换象限(左上与右下进行交换)
        Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        // 交换象限（右上与左下进行交换）
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }
    inline Mat dft(cv::Mat srcImage)
    {
        // 2 将输入图像扩展到最佳尺寸,边界用 0 补充
        int m = getOptimalDFTSize(srcImage.rows);
        int n = getOptimalDFTSize(srcImage.cols);
        opt_h = m;
        opt_w = n;

        // 将添加的像素初始化为 0
        Mat padded;
        copyMakeBorder(srcImage, padded, 0, m - srcImage.rows,
                       0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
        opt_size = padded.size();
        // 3 为傅里叶变换的结果(实部和虚部)分配存储空间
        // 将数组组合合并为一个多通道数组
        Mat planes[] = {Mat_<double>(padded), Mat::zeros(padded.size(), CV_64F)};
        Mat complexI;
        cv::merge(planes, 2, complexI);
        // 4 进行傅里叶变换
        cv::dft(complexI, complexI);
        // idft(complexI, complexI);
        // 将复数转换为幅值，即=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        // 将多通道数组分离为几个单通道数组
        this->dft_img = complexI.clone();
        cv::split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        cv::magnitude(planes[0], planes[1], planes[0]);
        Mat magImage = planes[0];
        // 对数尺度缩放
        magImage += Scalar::all(1);
        log(magImage, magImage); // 求自然对数
        // 剪切和重分布幅度图象限
        // 若有奇数行或奇数列，进行频谱剪裁
        magImage = magImage(Rect(0, 0, magImage.cols & -2, magImage.rows & -2));
        // 重新排列傅立叶图像中的象限，使得原点位于图像中心
        fftshift(magImage);
        // 归一化，用 0 到 1 的浮点值将矩阵变换为可视的图像格式
        normalize(magImage, magImage, 0, 1, NORM_MINMAX);
        return magImage;
    }
    inline static Mat butterworth_lbrf_kernel(cv::Size size, double sigma, int n, bool show_kernel = true)
    {
        Mat butterworth_low_pass(size, CV_64FC1); // ，CV_64FC1
        int rows = size.height;
        int cols = size.width;
        double D0 = sigma; // 半径 D0 越⼩，模糊越⼤；半径 D0 越⼤，模糊越⼩
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double d = sqrt(pow((i - rows / 2), 2) + pow((j - cols / 2), 2)); // 分⼦,计算 pow 必须为 double 型
                butterworth_low_pass.at<double>(i, j) = double(1.0 / (1 + pow(d / D0, 2 * n)));
            }
        }

        // putText(butterworth_low_pass, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
        if (show_kernel)
            imshow("Butterworth LOW PASS", butterworth_low_pass);
        return butterworth_low_pass;
    }
    inline static Mat guassus_bulr_kernel(cv::Size size, double sigma, bool show_kernel = true)
    {
        Mat guassus_bulr(size, CV_64FC1); // ，CV_64FC1
        int rows = size.height;
        int cols = size.width;
        double D0 = sigma; // 半径 D0 越⼩，模糊越⼤；半径 D0 越⼤，模糊越⼩
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double d = sqrt(pow((i - rows / 2), 2) + pow((j - cols / 2), 2)); // 分⼦,计算 pow 必须为 double 型
                guassus_bulr.at<double>(i, j) = double(exp(-pow(d, 2) / (2 * pow(D0, 2))));
            }
        }

        // putText(guassus_bulr, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
        if (show_kernel)
            imshow("Guassus BULR", guassus_bulr);
        return guassus_bulr;
    }
    inline static Mat ideal_bulr_kernel(cv::Size size, double sigma, bool show_kernel = true)
    {
        Mat ideal_bulr(size, CV_64FC1); // ，CV_64FC1
        int rows = size.height;
        int cols = size.width;
        double D0 = sigma; // 半径 D0 越⼩，模糊越⼤；半径 D0 越⼤，模糊越⼩
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double d = sqrt(pow((i - rows / 2), 2) + pow((j - cols / 2), 2)); // 分⼦,计算 pow 必须为
                if (d <= D0)
                    ideal_bulr.at<double>(i, j) = 1;
                else
                    ideal_bulr.at<double>(i, j) = 0;
            }
        }

        // putText(ideal_bulr, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
        if (show_kernel)
            imshow("Ideal BULR", ideal_bulr);
        return ideal_bulr;
    }
    inline static Mat ideal_hbrf_kernel(cv::Size size, double sigma, bool show_kernel = true)
    {
        Mat ideal_high_pass(size, CV_64FC1); // ，CV_64FC1
        int rows = size.height;
        int cols = size.width;
        double D0 = sigma; // 半径 D0 越⼩，模糊越⼤；半径 D0 越⼤，模糊越⼩
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double d = sqrt(pow((i - rows / 2), 2) + pow((j - cols / 2), 2)); // 分⼦,计算 pow 必须为 double 型
                if (d <= D0)
                    ideal_high_pass.at<double>(i, j) = 0;
                else
                    ideal_high_pass.at<double>(i, j) = 1;
            }
        }
        std::string name = "Ideal HIGH PASS d0=" + std::to_string(sigma);
        // putText(ideal_high_pass, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
        if (show_kernel)
            imshow("Ideal HIGH PASS", ideal_high_pass);
        return ideal_high_pass;
    }
    // 一些系统模型函数
    //   degradation model
    inline static Mat degradation_model_kernel(cv::Size size, double k, bool show_kernel = true)
    {
        Mat guassus_bulr(size, CV_64FC1); // ，CV_64FC1
        int rows = size.height;
        int cols = size.width;
        if (k <= 0)
            k = 0.00000000001;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double d = -k * pow(pow((i - rows / 2.0), 2) + pow((j - cols / 2.0), 2), 5.0 / 6.0); // 分⼦,计算 pow 必须为 double 型
                d = double(exp(d));
                guassus_bulr.at<double>(i, j) = d;
                // cout<<d<<endl;
            }
        }

        // putText(guassus_bulr, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
        if (show_kernel)
            imshow("degradation model", guassus_bulr);
        return guassus_bulr;
    }
    //  move model
    inline static Mat move_model_kernel(cv::Size size, double move_x, double move_y, double T, bool show_kernel = true)
    {
        Mat move_model(size, CV_64FC2); // ，CV_64FC1
        int rows = size.height;
        int cols = size.width;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                auto d = T * sinc(PI * (move_y * (i - rows / 2) + move_x * (j - cols / 2))) * exp(-1i * PI * (move_y * (i - rows / 2) + move_x * (j - cols / 2))); // 分⼦,计算 pow 必须为 double 型
                // cout<<d<<endl;
                move_model.at<cv::Vec2d>(i, j)[0] = d.real();
                move_model.at<cv::Vec2d>(i, j)[1] = d.imag();
            }
        }
        // putText(guassus_bulr, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
        if (show_kernel)
        {
            Mat planes[2];
            cv::split(move_model, planes);
            cv::magnitude(planes[0], planes[1], planes[0]);
            imshow("move model", planes[0]);
        }
        return move_model;
    }
    inline static Mat move_inv_filter_kernel(cv::Size size, double move_x, double move_y, double T, bool show_kernel = true)
    {
        Mat move_model(size, CV_64FC2); // ，CV_64FC1
        int rows = size.height;
        int cols = size.width;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                auto d = T * sinc(PI * (move_y * (i - rows / 2) + move_x * (j - cols / 2))) * exp(-1i * PI * (move_y * (i - rows / 2) + move_x * (j - cols / 2))); // 分⼦,计算 pow 必须为 double 型
                if (abs(d * d) < 0.00000001)
                {
                    d = 0.000001 + 0.000001i;
                }
                d /= d * d;
                move_model.at<cv::Vec2d>(i, j)[0] = d.real();
                move_model.at<cv::Vec2d>(i, j)[1] = d.imag();
            }
        }
        // putText(guassus_bulr, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
        if (show_kernel)
        {
            Mat planes[2];
            cv::split(move_model, planes);
            cv::magnitude(planes[0], planes[1], planes[0]);
            imshow("move model", planes[0]);
        }
        return move_model;
    }
    inline Mat apply_model(cv::Mat kernel)
    {
        Mat BLUR;
        if (kernel.channels() == 1)
        {
            Mat blur_r, blur_i;
            multiply(dft_img_real, kernel, blur_r);
            multiply(dft_img_imag, kernel, blur_i);

            Mat plane1[] = {blur_r, blur_i};
            merge(plane1, 2, BLUR);
        }
        else if (kernel.channels() == 2)
        {
            Mat planes[] = {Mat::zeros(kernel.size(), CV_64F), Mat::zeros(kernel.size(), CV_64F)};
            split(kernel, planes);

            Mat ac = dft_img_real.mul(planes[0]); // a*c
            Mat bd = dft_img_imag.mul(planes[1]); // b*d
            Mat bc = dft_img_imag.mul(planes[0]); // b*c
            Mat ad = dft_img_real.mul(planes[1]); // a*d

            Mat real = ac - bd; // real part of result
            Mat imag = bc + ad; // imaginary part of result

            Mat out[] = {real, imag};
            merge(out, 2, BLUR);
        }
        idft(BLUR, BLUR);
        Mat final_planes[] = {Mat::zeros(BLUR.size(), CV_64F), Mat::zeros(BLUR.size(), CV_64F)};
        split(BLUR, final_planes);
        magnitude(final_planes[0], final_planes[1], final_planes[0]);
        normalize(final_planes[0], final_planes[0], 0, 1, NORM_MINMAX);
        return final_planes[0];
    }

    inline static Mat wiener_filter_kernel(const cv::Mat& input, int kernel_size)
    {
        CV_Assert(input.channels() == 1);
        cv::Mat input64f;
        input.convertTo(input64f, CV_64F); // Convert input to CV_64F if it is not already

        cv::Mat local_mean;
        cv::Mat local_var;
        cv::boxFilter(input64f, local_mean, CV_64F, cv::Size(kernel_size, kernel_size));
        cv::sqrBoxFilter(input64f, local_var, CV_64F, cv::Size(kernel_size, kernel_size));

        local_var -= local_mean.mul(local_mean);
        double mean_var = cv::mean(local_var)[0];

        cv::Mat output;
        cv::normalize(local_var - mean_var, output, 0, 1, cv::NORM_MINMAX);
        output = local_mean + output.mul(input64f - local_mean);
        return output;

    } //  wiener_filter_kernel
    inline  static cv::Mat applyWienerFilter(cv::Mat& inputImage ,cv::Mat & kernel, double noisePower = 0.01)
    {

        cv::Mat outputImage, complexI, planes[2], wienerFilter;
        if (inputImage.type() == CV_8U) {
            inputImage.convertTo(inputImage, CV_64F);
        }
        // Compute DFT of input image
        cv::dft(inputImage, complexI, DFT_COMPLEX_OUTPUT);
        // Split Fourier in real and imaginary parts
        cv::split(complexI, planes);
        // Compute the magnitude
        cv::Mat mag, magSqr;
        cv::magnitude(planes[0], planes[1], mag);
        // Compute power spectrum (magnitude squared)
        cv::pow(kernel, 2, magSqr);
        // cv::normalize(magSqr, magSqr, 0, 1, NORM_MINMAX);
        // Create Wiener filter
          // Noise power, to be set as per the noise in the image
        wienerFilter = magSqr / (magSqr + noisePower);
        // Apply Wiener filter in frequency domain
        planes[0] = planes[0].mul(wienerFilter);
        planes[1] = planes[1].mul(wienerFilter);
        // Merge and inverse DFT
        cv::merge(planes, 2, complexI);
        cv::idft(complexI, complexI);
        // Split Fourier in real and imaginary parts
        cv::split(complexI, planes);
        // Compute the magnitude and convert to CV_8U
        cv::magnitude(planes[0], planes[1], mag);
        cv::normalize(mag, outputImage, 0, 1, NORM_MINMAX);
        //mag.convertTo(outputImage, CV_8U);

        return outputImage;
    }

 
};
