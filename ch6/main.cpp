#include <iostream>
#include <opencv2/opencv.hpp>
#include "img_trans.h"
#include "utils.hpp"
#include "filter.h"
#include "Freq_domain.hpp"
#define ITEM 6

#ifdef _WIN64 || _WIN32
#define IMG_PATH "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch6\\PeppersRGB.bmp"
#define IMG_PATH2 "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch6\\hailang.png"
#define IMG_PATH3 "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch6\\lena.jpg"
#else
#define IMG_PATH "../PeppersRGB.bmp"
#define IMG_PATH2 "../hailang.png"
#endif

int main(int arvc, char *argv[])
{
#if ITEM == 0 || ITEM == 1 || ITEM == 2 || ITEM == 3
    Mat img = imread(IMG_PATH3, 0);
    if (img.empty())
    {
        cout << "img is empty" << endl;
        return -1;
    }
    Noise noise(img, "saltandpepper");
    Mat img_noise_salt = noise.saltandpepper(0.15);
    Mat img_noise_normal = noise.normal(0, 40);
    Mat img_noise_gaussian = noise.gaussian(0, 15);
    imshow("img", img);
    imshow("img_noise_salt", img_noise_salt);
    imshow("img_noise_normal", img_noise_normal);
    imshow("img_noise_gaussian", img_noise_gaussian);
    waitKey(0);
#endif
#if ITEM == 1
    // 尝试对高斯噪声进行滤除
    vector<cv::Mat> res;
    vector<string> types;
    res.reserve(5);
    Filter filter(img_noise_gaussian);
    res.push_back(filter.meanFilter(3));
    types.push_back("mean:3");
    res.push_back(filter.meanFilter(5));
    types.push_back("mean:5");
    res.push_back(filter.medianFilter(3));
    types.push_back("median:3");
    res.push_back(filter.medianFilter(5));
    types.push_back("median:5");
    res.push_back(filter.gaussianFilter(3, 20));
    types.push_back("gaussian:3");
    res.push_back(filter.gaussianFilter(5, 20));
    types.push_back("gaussian:5");
    res.push_back(filter.bilateralFilter(3, 5));
    types.push_back("bilateral:3");
    res.push_back(filter.bilateralFilter(5, 5));
    types.push_back("bilateral:5");
    res.push_back(filter.Geometric_mean_filter(3));
    types.push_back("Geometric_mean_filter:3");
    res.push_back(filter.Geometric_mean_filter(5));
    types.push_back("Geometric_mean_filter:5");
    res.push_back(filter.Harmonic_mean_filter(3));
    types.push_back("Harmonic_mean_filter:3");
    res.push_back(filter.Harmonic_mean_filter(5));
    types.push_back("Harmonic_mean_filter:5");
    // iterative
    for (int i = 0; i < res.size(); i++)
    {
        imshow(types[i], res[i]);
        cv::waitKey(0);
    }

#elif ITEM == 2
    // 尝试 椒盐噪声进行滤除
    vector<cv::Mat> res;
    vector<string> types;
    res.reserve(10);
    Filter filter(img_noise_salt);
    res.push_back(filter.meanFilter(3));
    types.push_back("mean:3");
    res.push_back(filter.meanFilter(5));
    types.push_back("mean:5");
    res.push_back(filter.medianFilter(3));
    types.push_back("median:3");
    res.push_back(filter.medianFilter(5));
    types.push_back("median:5");
    res.push_back(filter.gaussianFilter(3, 20));
    types.push_back("gaussian:3");
    res.push_back(filter.gaussianFilter(5, 20));
    types.push_back("gaussian:5");
    res.push_back(filter.bilateralFilter(3, 5));
    types.push_back("bilateral:3");
    res.push_back(filter.bilateralFilter(5, 5));
    types.push_back("bilateral:5");
    res.push_back(filter.Geometric_mean_filter(3));
    types.push_back("Geometric_mean_filter:3");
    res.push_back(filter.Geometric_mean_filter(5));
    types.push_back("Geometric_mean_filter:5");
    res.push_back(filter.Harmonic_mean_filter(3));
    types.push_back("Harmonic_mean_filter:3");
    res.push_back(filter.Harmonic_mean_filter(5));
    types.push_back("Harmonic_mean_filter:5");
    res.push_back(filter.Contraharmonic_mean_filter(3, 1.5));
    types.push_back("Contraharmonic_mean_filter:3");
    res.push_back(filter.Contraharmonic_mean_filter(5, 1.5));
    types.push_back("Contraharmonic_mean_filter:5");
    res.push_back(filter.Contraharmonic_mean_filter(3, -1.5));
    types.push_back("Contraharmonic_mean_filter:-3");
    res.push_back(filter.Contraharmonic_mean_filter(5, -1.5));
    types.push_back("Contraharmonic_mean_filter:-5");
    // iterative
    for (int i = 0; i < res.size(); i++)
    {
        cv::imshow(types[i], res[i]);
        cv::waitKey(0);
    }
#elif ITEM == 3
    vector<cv::Mat> res;
    vector<string> types;
    res.reserve(10);
    Filter filter(img_noise_normal);
    res.push_back(filter.meanFilter(3));
    types.push_back("mean:3");
    res.push_back(filter.meanFilter(5));
    types.push_back("mean:5");
    res.push_back(filter.medianFilter(3));
    types.push_back("median:3");
    res.push_back(filter.medianFilter(5));
    types.push_back("median:5");
    res.push_back(filter.gaussianFilter(3, 20));
    types.push_back("gaussian:3");
    res.push_back(filter.gaussianFilter(5, 20));
    types.push_back("gaussian:5");
    res.push_back(filter.bilateralFilter(3, 5));
    types.push_back("bilateral:3");
    res.push_back(filter.bilateralFilter(5, 5));
    types.push_back("bilateral:5");
    res.push_back(filter.Geometric_mean_filter(3));
    types.push_back("Geometric_mean_filter:3");
    res.push_back(filter.Geometric_mean_filter(5));
    types.push_back("Geometric_mean_filter:5");
    res.push_back(filter.Harmonic_mean_filter(3));
    types.push_back("Harmonic_mean_filter:3");
    res.push_back(filter.Harmonic_mean_filter(5));
    types.push_back("Harmonic_mean_filter:5");
    res.push_back(filter.Contraharmonic_mean_filter(3, 1.5));
    types.push_back("Contraharmonic_mean_filter:3");
    res.push_back(filter.Contraharmonic_mean_filter(5, 1.5));
    types.push_back("Contraharmonic_mean_filter:5");
    res.push_back(filter.Contraharmonic_mean_filter(3, -1.5));
    types.push_back("Contraharmonic_mean_filter:-3");
    res.push_back(filter.Contraharmonic_mean_filter(5, -1.5));
    types.push_back("Contraharmonic_mean_filter:-5");
    // iterative
    for (int i = 0; i < res.size(); i++)
    {
        imshow(types[i], res[i]);
        cv::waitKey(0);
    }
#elif ITEM == 4
    Mat img = cv::imread(IMG_PATH, 0);
    cv::imshow("img", img);
    Freq_domain freq_ana(img);
    cv::imshow("dft_img_mag", freq_ana.dft_img_to_show);
    Mat move_burl1 = Freq_domain::degradation_model_kernel(freq_ana.opt_size, 0.0025);
    waitKey(0);
    Mat move_burl2 = Freq_domain::degradation_model_kernel(freq_ana.opt_size, 0.00025);
    waitKey(0);
    Mat move_burl3 = Freq_domain::degradation_model_kernel(freq_ana.opt_size, 0.000025);
    waitKey(0);
    Mat move_burl4 = Freq_domain::degradation_model_kernel(freq_ana.opt_size, 0.0000025);
    waitKey(0);
    Mat move_img = freq_ana.apply_model(move_burl1);
    cv::imshow("move_img", move_img);
    Mat move_img2 = freq_ana.apply_model(move_burl2);
    cv::imshow("move_img2", move_img2);
    Mat move_img3 = freq_ana.apply_model(move_burl3);
    cv::imshow("move_img3", move_img3);
    Mat move_img4 = freq_ana.apply_model(move_burl4);
    cv::imshow("move_img4", move_img4);
    cv::waitKey(0);
#elif ITEM == 5
    Mat img = cv::imread(IMG_PATH, 0);
    cv::imshow("img", img);
    Freq_domain freq_ana(img);
    cv::imshow("dft_img_mag", freq_ana.dft_img_to_show);
    Mat move_burl = Freq_domain::move_model_kernel(freq_ana.opt_size, 0.050, 0, 1);

    // Mat move_burl = Freq_domain::guassus_bulr_kernel(freq_ana.opt_size, 80);
    // cv::imshow("dft_img_phase",freq_ana.dft_img_phase_show);
    // Mat move_burl =Freq_domain::move_model_kernel(freq_ana.opt_size,50,0,5 );
    // Mat move_burl = Freq_domain::guassus_bulr_kernel(freq_ana.opt_size,80);
    // Mat move_burl = Freq_domain::butterworth_lbrf_kernel(freq_ana.opt_size, 80, 10);
    // Mat move_burl = Freq_domain::ideal_bulr_kernel(freq_ana.opt_size, 200);

    Mat move_img = freq_ana.apply_model(move_burl);

    cv::imshow("move img", move_img);
    Freq_domain inv_filter(move_img);
    cv::imshow(" move dft_img_mag", inv_filter.dft_img_to_show);
    cv::waitKey(0);

    Mat inv_burl = inv_filter.move_inv_filter_kernel(freq_ana.opt_size, 0.05, 0.00, 1);
    Mat inv_img = inv_filter.apply_model(inv_burl);
    cv::imshow("inv_img", inv_img);
    Freq_domain inv_res(inv_img);
    cv::imshow("inv after mag", inv_res.dft_img_to_show);
    cv::waitKey(0);
    // int kernel_size = int(0.05 * img.cols);
    // cv::Mat kernel = cv::Mat::zeros(1, kernel_size, CV_32F);
    // kernel = 1.0 / kernel_size;
    // cv::Mat blurred;
    // cv::filter2D(img, blurred, -1, kernel);
    // imshow("blured" ,blurred);
    //   inline static Mat wiener_filter_kernel(const cv::Mat &input, int kernel_size)
    //     {
    //         CV_Assert(input.channels() == 1);
    //         cv::Mat local_mean;
    //         cv::Mat local_var;
    //         cv::boxFilter(input, local_mean, CV_64F, cv::Size(kernel_size, kernel_size));
    //         cv::sqrBoxFilter(input, local_var, CV_64F, cv::Size(kernel_size, kernel_size));
    //         local_var -= local_mean.mul(local_mean);
    //         double mean_var = cv::mean(local_var)[0];
    //         cv::Mat output;
    //         cv::normalize(local_var - mean_var, output, 0, 1, cv::NORM_MINMAX);
    //         output = local_mean + output.mul(input - local_mean);
    //         return output;
    //     }

#elif ITEM == 6

    Mat img = cv::imread(IMG_PATH, 0);
    cv::imshow("img", img);
    Mat img2 = cv::imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\ch6\\move_img_0.0025.png", 0);

    cv::imshow("ima2", img2);
    // 维纳滤波
    cv::namedWindow("result");

    cv::createTrackbar(
        "ksize", "result", 0, 10000, [](int pos, void *userdata)
        { 
            if (pos == 0) pos = 1;
            
            cv::Mat ima2 = *(Mat*)userdata;
            Freq_domain inv_filter(ima2);
            Mat inv_burl = inv_filter. wiener_filter_kernel(ima2, pos/1000);
            Mat wiener =    inv_filter.apply_model(inv_burl);
            cv::imshow("wiener", wiener); },
        &img2);
    waitKey(0);
    // 逆滤波
#endif
    return 0;
}