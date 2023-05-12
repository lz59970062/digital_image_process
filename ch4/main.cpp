#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"
#include "histogram.hpp"

#define ITEM 2
#ifdef _WIN64||_WIN32
#define IMG_PATH "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch4\\PeppersRGB.bmp"
#else
#define IMG_PATH "../PeppersRGB.bmp"
#endif

int main(int arvc, char *argv[])
{
    Mat img;
#if ITEM == 1

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
#else ITEM == 2
 
    img = imread(IMG_PATH, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    cv::imshow("imgraw", img);
    Histogram1D h1;
    Mat img2 = h1.getHistogramImage(img);
    cv::imshow("imgraw hismap", img2);
    Mat img_log,img_pow,img_eq;
    img.convertTo(img, CV_32F);
    
    img+=1;
    img/=255;
    cv::log(img , img_log);
    cv::pow(img, 1.5, img_pow);
    // img_log*=255;
    
	normalize(img_log,img_log, 0, 255, NORM_MINMAX);
    normalize(img_pow, img_pow, 0, 255, NORM_MINMAX);
    img_log.convertTo(img_log, CV_8U);
    img_pow.convertTo(img_pow, CV_8U);

    cv::imshow("imgraw log", img_log);
    Mat img4 = h1.getHistogramImage(img_log);
    cv::imshow("imgraw log hismap", img4);

    cv::imshow("imgraw pow", img_pow);
    Mat img5 = h1.getHistogramImage(img_pow);
    cv::imshow("imgraw pow hismap", img5);
    cv::waitKey(0);

#endif
    return 0;
}
