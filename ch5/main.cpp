#include <iostream>
#include <opencv2/opencv.hpp>
#include "img_trans.h"
#include "utils.hpp"
#include "filter.h"
#include "Freq_domain.hpp"

#define ITEM 9
#ifdef _WIN64 || _WIN32
#define IMG_PATH "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch5\\PeppersRGB.bmp"
#define IMG_PATH2 "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch5\\lena.jpg"
#else
#define IMG_PATH "../PeppersRGB.bmp"
#define IMG_PATH2 "../hailang.png"
#endif
using namespace std;
using namespace cv;

int main(int arvc, char *argv[])
{
#if ITEM == 1
    Mat img = imread(IMG_PATH, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    cv::imshow("raw", img);
     waitKey(0);
    Enhancement enh(img);
    Histogram hist; 
    
    std::vector<vector<double>> piece_a = {{0, 100, 120, 255}, {0, 100, 120, 200}};
    std::vector<vector<double>> piece_b = {{0, 150, 160, 255}, {0, 120, 150, 255}};
    std::vector<cv::Mat> piece_res;
    std::vector<cv::Mat> piece_hist_res;
    piece_res.reserve(2);
    for (int i = 0; i < piece_a.size(); i++)
    {
        cv::Mat tmp = enh.piecewiseLinearTransform(piece_a[i], piece_b[i]);
        auto  piece_hist_tmp = hist.computeHistogram(tmp);
        hist.drawHistogram(piece_hist_tmp, "piece_hist_tmp_hist");
        cv::imshow("linear", tmp);
        waitKey(0); 
    }
   
    waitKey(0);
    cv::resize(img, img, cv::Size(img.rows / 2, img.cols / 2));
    // cv::imshow("img21",img);

    Enhancement enh2(img);
    std::vector<cv::Mat> gamma_res;
    for (double i = 0.2; i < 3; i += 0.2)
    {
        cv::Mat tmp = enh2.gammaTransform(i);
        cv::putText(tmp, "gamma:" + std::to_string(i), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
        gamma_res.push_back(tmp);
    }
    hconcat(gamma_res, enh2.result_img);
    imshow("gamma", enh2.result_img);
    waitKey(0);
    std::vector<cv::Mat> log_res;
    for (double i = 0.5; i < 1.5; i += 0.2)
    {
        cv::Mat tmp = enh2.logTransform(i);
        cv::putText(tmp, "log:" + std::to_string(i), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
        log_res.push_back(tmp);
    }
    hconcat(log_res, enh2.result_img);
    imshow("log", enh2.result_img);
    waitKey(0);

#elif ITEM == 3

    Mat img = imread(IMG_PATH);
    // cv::resize(img,img,cv::Size(img.rows/2,img.cols/2), 0, 0, cv::INTER_LINEAR);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    cv::imshow("img", img);
    cv::waitKey(0);
    vector<Mat> bgr_planes;
    bgr_planes.reserve(3);
    split(img, bgr_planes);
    Enhancement enhb(bgr_planes[0]), enhg(bgr_planes[1]), enhr(bgr_planes[2]);
    cv::Mat tmpb = enhb.logTransform(0.5);
    cv::Mat tmpg = enhb.logTransform(0.5);
    cv::Mat tmpr = enhr.logTransform(0.5);
    std::vector<Mat> bgr_res = {tmpb, tmpg, tmpr};
    cv::imshow("img2", img);
    Histogram hist;
    std::vector<Mat> hist_res;
    hist_res = hist.computeHistogram(img);
    hist.drawHistogram(hist_res, "dark image");
    cv::waitKey(0);

#elif ITEM == 2
    Mat img = imread(IMG_PATH);
    // cv::resize(img,img,cv::Size(img.rows/2,img.cols/2), 0, 0, cv::INTER_LINEAR);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    // imshow("img", img);
    // waitKey(0);

    // 将图片进行灰度变换，增加暗区
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(
        "log", "img", 0, 400, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Histogram hist;
            auto hist_raw = hist.computeHistogram(gray_img);
            hist.drawHistogram(hist_raw, "raw_hist");
            Enhancement  enhgray(gray_img);
            cv::Mat tmpgray = enhgray.logTransform(pos/100);
            cv::imshow("img", tmpgray); 
            auto hist_tmp = hist.computeHistogram(tmpgray);
            hist.drawHistogram(hist_tmp, "log_hist"); },
        &img);
    cv::createTrackbar(
        "pow", "img", 0, 400, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Histogram hist;
            auto hist_raw = hist.computeHistogram(gray_img);
            hist.drawHistogram(hist_raw, "raw_hist");
            Enhancement  enhgray(gray_img);
            cv::Mat tmpgray = enhgray.gammaTransform(pos/100);
            cv::imshow("img", tmpgray); 
            auto hist_tmp = hist.computeHistogram(tmpgray);
            hist.drawHistogram(hist_tmp, "gammaTransform_hist"); },
        &img);
    waitKey(0);
#elif ITEM == 4
    // Mat img = imread(IMG_PATH2,0);
    Mat img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\ch5\\dark.png");
    imshow("raw", img);
    Histogram hist;
    auto hist_raw = hist.computeHistogram(img);
    hist.drawHistogram(hist_raw, "raw_hist");
    Mat equal_img;
    equal_img = HistogramEqualizer::equalize(img);
    Histogram hist_equal;
    auto hist_equal_res = hist_equal.computeHistogram(equal_img);
    hist_equal.drawHistogram(hist_equal_res, "equal_hist");

    imshow("equal_img", equal_img);
    waitKey(0);
    waitKey(0);
  
    
#elif ITEM == 5

    Mat img = imread(IMG_PATH);
    // cv::resize(img,img,cv::Size(img.rows/2,img.cols/2), 0, 0, cv::INTER_LINEAR);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }

    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(
        "mean_ksize", "img", 0,200, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;

            if(pos==0) pos=1;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Histogram hist;
            auto hist_raw = hist.computeHistogram(gray_img);
            hist.drawHistogram(hist_raw, "raw_hist");
            // Enhancement  enhgray(gray_img);
            // cv::Mat tmpgray = enhgray.logTransform(pos/100);
            Filter filter(img);
            cv::Mat tmpgray = filter.meanFilter(pos);
            cv::imshow("img", tmpgray); 
            auto hist_tmp = hist.computeHistogram(tmpgray);
            hist.drawHistogram(hist_tmp, "mean_hist"); },
        &img);
    waitKey(0);
    cv::createTrackbar(
        "median_ksize", "img", 0, 200, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            if(pos==0) pos=1;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Histogram hist;
            auto hist_raw = hist.computeHistogram(gray_img);
            hist.drawHistogram(hist_raw, "raw_hist");
            // Enhancement  enhgray(gray_img);
            // cv::Mat tmpgray = enhgray.logTransform(pos/100);
            Filter filter(img);
            cv::Mat tmpgray = filter.medianFilter(pos);
            cv::imshow("img", tmpgray); 
            auto hist_tmp = hist.computeHistogram(tmpgray);
            hist.drawHistogram(hist_tmp, "median_hist"); },
        &img);
    waitKey(0);
 

#elif ITEM == 6
    Mat img = imread(IMG_PATH2, 0);
    img.convertTo(img, CV_64F);
    img /= 255;
    imshow("raw", img);
    Mat result_img;
    Mat sobel_kernel1 = Filter::gen_sobel_kernel(0);
    Mat sobel_kernel2 = Filter::gen_sobel_kernel(1);
    Mat laplacian_kernel = Filter::gen_laplacian_kernel(3);
    Mat prewitt_kernel1 = Filter::gen_prewitt_kernel(0);
    Mat prewitt_kernel2 = Filter::gen_prewitt_kernel(1);
    Mat roberts_kernel1 = Filter::gen_roberts_kernel(0);
    Mat roberts_kernel2 = Filter::gen_roberts_kernel(1);
    Mat result_img_sobel;

    cv::filter2D(img, result_img, -1, sobel_kernel1);
    imshow("sobel1", abs(result_img));
    imshow("sobel11", (result_img)+0.2);

    cv::filter2D(img, result_img_sobel, -1, sobel_kernel2);
    imshow("sobel2",  result_img_sobel +0.2);

    cv::filter2D(result_img, result_img, -1, sobel_kernel2);
    imshow("sobel 1 and 2", abs(result_img));
    waitKey(0);
      
    // imshow("sobel 1 and 2",result_img );
    cv::Mat result_img_laplacian;
    cv::filter2D(img, result_img_laplacian, -1, laplacian_kernel);
    imshow("laplacia", result_img_laplacian+0.2);
    cv::Mat result_img_prewitt;
    cv::filter2D(img, result_img_prewitt, -1, prewitt_kernel1);
    imshow("prewitt1", result_img_prewitt+0.2);
    cv::Mat result_img_prewitt2;
    cv::filter2D(img, result_img_prewitt2, -1, prewitt_kernel2);
    imshow("prewitt2", result_img_prewitt2+0.2);
    cv::filter2D(result_img_prewitt, result_img_prewitt, -1, prewitt_kernel2);
    imshow("prewitt 1 and 2", result_img_prewitt+0.2);
    cv::Mat result_img_roberts;
    cv::filter2D(img, result_img_roberts, -1, roberts_kernel1);
    imshow("roberts1", result_img_roberts+0.2);
    cv::Mat result_img_roberts2;
    cv::filter2D(img, result_img_roberts2, -1, roberts_kernel2);
    imshow("roberts2", result_img_roberts2+0.2);
    cv::filter2D(result_img_roberts, result_img_roberts, -1, roberts_kernel2);
    imshow("roberts 1 and 2", result_img_roberts+0.2);
    waitKey(0);
    Filter filter(img);
    Histogram hist;
    Mat img2 = img * 255;
    img2.convertTo(img2, CV_8U);
    auto hist_img = hist.computeHistogram(img2);
    hist.drawHistogram(hist_img, "hist_img ");
    // 交叉微分算子处理
    cv::Mat roberts_result = filter.roberts_enhence(1);
    cv::imshow("roberts_result ", roberts_result);
    roberts_result *= 255;
    roberts_result.convertTo(roberts_result, CV_8U);
    auto hist_roberts = hist.computeHistogram(roberts_result);
    hist.drawHistogram(hist_roberts, "roberts_hist");
    // 梯度法处理
    cv::Mat prewitt_result = filter.prewitt_enhence(1);
    cv::imshow("prewitt_result ", prewitt_result);
    prewitt_result *= 255;
    prewitt_result.convertTo(prewitt_result, CV_8U);
    auto hist_prewitt = hist.computeHistogram(prewitt_result);
    hist.drawHistogram(hist_prewitt, "prewitt_hist");
    // Sobel 算子处理
    cv::Mat sobel_result = filter.sobel_enhence(1);
    cv::imshow("sobel_result ", sobel_result);
    sobel_result *= 255;
    sobel_result.convertTo(sobel_result, CV_8U);
    auto hist_sobel = hist.computeHistogram(sobel_result);
    hist.drawHistogram(hist_sobel, "sobel_hist");
    // 二阶拉普拉斯算子处理
    cv::Mat laplace_result = filter.laplace_enhence(3); // 假设使用3x3的拉普拉斯核
    cv::imshow("laplace_result ", laplace_result);
    laplace_result *= 255;
    laplace_result.convertTo(laplace_result, CV_8U);
    auto hist_laplace = hist.computeHistogram(laplace_result);
    hist.drawHistogram(hist_laplace, "laplace_hist");
    waitKey(0);
        waitKey(0);
            waitKey(0);
            waitKey(0);
            waitKey(0);
#elif ITEM == 7
    Mat img = imread(IMG_PATH2, 0);
    img.convertTo(img, CV_64F);
    Freq_domain freq(img);
    imshow("def img", freq.dft_img_to_show);
    img /= 255;
    imshow("img", img);
    // waitKey(0);
    // Mat real_toshow,image_toshow;
    //对数缩放
    // freq.dft_img_real += 1;
    // freq.dft_img_imag += 1;
    // cv::log(freq.dft_img_real, freq.dft_img_real);
    // cv::log(freq.dft_img_imag, freq.dft_img_imag);
    // cv::normalize(freq.dft_img_real, real_toshow, 0, 1, NORM_MINMAX);
    // cv::normalize(freq.dft_img_imag, image_toshow, 0, 1, NORM_MINMAX);
    // imshow("real", real_toshow);
    // imshow("image", image_toshow);
    // waitKey(0);
    //  设置虚部为 0
    // freq.dft_img_imag = Mat::zeros(img.size(), CV_64F);
    // Mat planes[2] = {freq.dft_img_real, freq.dft_img_imag};
    // cv::merge(planes, 2, freq.dft_img);
    // cv::idft(freq.dft_img, freq.dft_img);
    // cv::split(freq.dft_img, planes);
    // cv::magnitude(planes[0], planes[1], freq.dft_img);
    // cv::normalize(freq.dft_img, freq.dft_img, 0, 1, NORM_MINMAX);
    // imshow("real img", freq.dft_img);
    // waitKey(0);
            
    Mat mag;
    cv::magnitude(freq.dft_img_real, freq.dft_img_imag, mag);
    Mat planes[2] = {freq.dft_img_real / mag, freq.dft_img_imag / mag};
    cv::merge(planes, 2, freq.dft_img);
    cv::idft(freq.dft_img, freq.dft_img);
    cv::split(freq.dft_img, planes);
    cv::magnitude(planes[0], planes[1], freq.dft_img);
    cv::normalize(freq.dft_img, freq.dft_img, 0, 1, NORM_MINMAX);
    imshow("real img", freq.dft_img);
    waitKey(0);
    waitKey(0);
            waitKey(0);
                waitKey(0);
#elif ITEM == 8

    Mat img = imread(IMG_PATH);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(
        "butterworth_lbrf", "img", 0, 256, [](int pos, void *userdata)
        {
            if(pos==0) pos=1;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Freq_domain freq(gray_img);
            Mat model= Freq_domain::butterworth_lbrf_kernel(freq.opt_size,pos,5,true);
            Mat tmpgray=freq.apply_model(model);
            cv::imshow("img", tmpgray); },
        &img);
    cv::createTrackbar(
        "ideal_bulr", "img", 0, 256, [](int pos, void *userdata)
        {
            if(pos==0) pos=1;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Freq_domain freq(gray_img);
            Mat model= Freq_domain::ideal_bulr_kernel(freq.opt_size,pos,true);
            Mat tmpgray=freq.apply_model(model);
            cv::imshow("img", tmpgray); },
        &img);
    cv::createTrackbar(
        "ideal_hpf", "img", 0, 256, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            if(pos==0) pos=1;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Freq_domain freq(gray_img);
            Mat model= Freq_domain::ideal_hbrf_kernel(freq.opt_size,pos,true);
            Mat tmpgray=freq.apply_model(model);
            cv::imshow("img", tmpgray); },
        &img);
    waitKey(0);

#elif ITEM == 9
    Mat img = imread(IMG_PATH2, 0);
    img.convertTo(img, CV_64F);
    Freq_domain freq(img);
    imshow("def img", freq.dft_img_to_show);
    img /= 255;
    imshow("img", img);
    waitKey(0);
    //  设置虚部为 0
    Mat mask(freq.opt_size, CV_64F, cv::Scalar(0));
    Mat mask2(freq.opt_size, CV_64F, cv::Scalar(0));

    mask.at<double>(freq.opt_h/3,freq.opt_w/2) = 100;
    mask.at<double>(freq.opt_h*2/3 ,freq.opt_w/2) = 100;

    freq.dft_img_real = freq.dft_img_real.mul(mask);
    freq.dft_img_imag = freq.dft_img_imag.mul(mask2);

    Mat planes[2] = {freq.dft_img_real, freq.dft_img_imag};
    cv::merge(planes, 2, freq.dft_img);
    cv::idft(freq.dft_img, freq.dft_img);
    cv::split(freq.dft_img, planes);
    cv::magnitude(planes[0], planes[1], freq.dft_img);
    cv::normalize(freq.dft_img, freq.dft_img, 0, 1, NORM_MINMAX);
    imshow("real img", freq.dft_img);
    waitKey(0);
    
#endif
    return 0;
}
