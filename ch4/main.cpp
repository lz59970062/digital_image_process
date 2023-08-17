#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "img_trans.h"
#define ITEM 3
#ifdef _WIN64 || _WIN32
#define IMG_PATH "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch4\\PeppersRGB.bmp"
#else
#define IMG_PATH "../PeppersRGB.bmp"
#endif

int main(int arvc, char *argv[])
{

#if ITEM == 1
    Mat img;
    VideoCapture cap(2);
    while (1)
    {
        cap >> img;
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        if (img.empty())
        {
            cout << "can not load image" << endl;
            return -1;
        }
        cv::imshow("imgraw", img);
        Histogram1D h1;
        Mat img2 = h1.getHistogramImage(img);
        cv::imshow("imgraw hismap", img2);
        // res2.push_back(img2);
        Mat img3 = h1.equalize(img);
        cv::imshow("imgeq", img3);
        Mat img4 = h1.getHistogramImage(img3);
        cv::imshow("imgeqhismap", img4);
        int q = cv::waitKey(3);
        if (q > 30)
            break;
    }
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
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Histogram hist;
            auto hist_raw = hist.computeHistogram(gray_img);
            hist.drawHistogram(hist_raw, "raw_hist");
            Enhancement  enhgray(gray_img);
            cv::Mat tmpgray = enhgray.logTransform(pos/100.0);
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
            cv::Mat tmpgray = enhgray.gammaTransform(pos/100.0);
            cv::imshow("img", tmpgray); 
            auto hist_tmp = hist.computeHistogram(tmpgray);
            hist.drawHistogram(hist_tmp, "gammaTransform_hist"); },
        &img);
    waitKey(0);
    waitKey(0);
    waitKey(0);
    waitKey(0);
    waitKey(0);
    waitKey(0);
    waitKey(0);
    waitKey(0);
    waitKey(0);

#elif ITEM == 3
    std::string img_path = "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch4\\dark.png";
    std::string img_path2 = "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch4\\light.png";
    std::string img_path3 = "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch4\\low_diff.png";
    Mat img = imread(img_path3);
    // cv::resize(img,img,cv::Size(img.rows/2,img.cols/2), 0, 0, cv::INTER_LINEAR);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(
        "log", "img", 0, 400, [](int pos, void *userdata)
        {
            cv::Mat img = *(Mat *)userdata;
            cv::Mat gray_img;
            cv::cvtColor(img,gray_img,cv::COLOR_BGR2GRAY);
            imshow("raw",gray_img);
            Histogram hist;
            auto hist_raw = hist.computeHistogram(gray_img);
            hist.drawHistogram(hist_raw, "raw_hist");
            Enhancement  enhgray(gray_img);
            cv::Mat tmpgray = enhgray.logTransform(pos/100.0);
            cv::imshow("img", tmpgray);
            auto hist_tmp = hist.computeHistogram(tmpgray);
            hist.drawHistogram(hist_tmp, "log_hist"); },
        &img);
    waitKey(0);
#endif
    return 0;
}
