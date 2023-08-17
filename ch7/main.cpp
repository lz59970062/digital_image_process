#include <iostream>
#include <opencv2/opencv.hpp>
#include "img_trans.h"
#include "utils.hpp"
#include "img_proc.hpp"
#include "filter.h"
#define ITEM 6
#ifdef _WIN64 || _WIN32
#define IMG_PATH "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch5\\PeppersRGB.bmp"
#define IMG_PATH2 "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch7\\lena.jpg"
#define IMG_PATH3 "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch7\\face.png"
#else
#define IMG_PATH "../PeppersRGB.bmp"
#define IMG_PATH2 "../hailang.png"
#define IMG_PATH2 "../lena.jpg"
#define IMG_PATH3 "../lena.jpg"
#endif
using namespace std;
using namespace cv;
int hMin = 0, sMin = 0, vMin = 0;
int hMax = 255, sMax = 255, vMax = 255;

cv::Rect roi;
bool selectObject = false;

void onMouse(int event, int x, int y, int, void *)
{
    if (selectObject)
    {
        roi.x = std::min(x, roi.x);
        roi.y = std::min(y, roi.y);
        roi.width = std::abs(x - roi.x);
        roi.height = std::abs(y - roi.y);
    }
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        roi = cv::Rect(x, y, 0, 0);
        selectObject = true;
        break;
    case cv::EVENT_LBUTTONUP:
        selectObject = false;
        break;
    }
}

void updateTrackbarPos(int, void *)
{
    hMin = cv::getTrackbarPos("Low H", "Trackbars");
    sMin = cv::getTrackbarPos("Low S", "Trackbars");
    vMin = cv::getTrackbarPos("Low V", "Trackbars");
    hMax = cv::getTrackbarPos("High H", "Trackbars");
    sMax = cv::getTrackbarPos("High S", "Trackbars");
    vMax = cv::getTrackbarPos("High V", "Trackbars");
}
int main(int arvc, char *argv[])
{
#if ITEM == 1
    Mat img = imread(IMG_PATH);
    // cv::resize(img,img,cv::Size(img.rows/2,img.cols/2), 0, 0, cv::INTER_LINEAR);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    // 将图片进行灰度变换，增加暗区
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(
        "Fix H", "img", 0, 200, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat imgr, imgg, imgb;
            Cimage_proc::color2rgb(img, imgr, imgg, imgb);
            // imshow("img",img);
            cv::imshow("imgr", imgr);
            cv::imshow("imgg", imgg);
            cv::imshow("imgb", imgb);
            cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
            imgr.convertTo(imgr, CV_64FC1);
            imgg.convertTo(imgg, CV_64FC1);
            imgb.convertTo(imgb, CV_64FC1);
            Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows, false);
            cv::Mat temp(img.size(), CV_64F, cv::Scalar(pos / 200.0));
            imgh=temp;
            cv::imshow("imgh", imgh);
            cv::imshow("imgs", imgs);
            cv::imshow("imgi", imgi);
            cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
            Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
            imgr2.convertTo(imgr2, CV_8UC1);
            imgg2.convertTo(imgg2, CV_8UC1);
            imgb2.convertTo(imgb2, CV_8UC1);
            cv::Mat img2(img.size(), CV_8UC3);
            vector<cv::Mat> channels;
            channels.push_back(imgb2);
            channels.push_back(imgg2);
            channels.push_back(imgr2);
            cv::merge(channels, img2);
            cv::imshow("img", img2); },
        &img);
    cv::createTrackbar(
        "Fix S", "img", 0, 200, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat imgr, imgg, imgb;
            Cimage_proc::color2rgb(img, imgr, imgg, imgb);
            // imshow("img",img);
            cv::imshow("imgr", imgr);
            cv::imshow("imgg", imgg);
            cv::imshow("imgb", imgb);
            cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
            imgr.convertTo(imgr, CV_64FC1);
            imgg.convertTo(imgg, CV_64FC1);
            imgb.convertTo(imgb, CV_64FC1);
            Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
            cv::Mat temp(img.size(), CV_64F, cv::Scalar(pos / 200.0));
            imgs=temp;
            cv::imshow("imgh", imgh);
            cv::imshow("imgs", imgs);
            cv::imshow("imgi", imgi);
            cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
            Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
            imgg2.convertTo(imgg2, CV_8UC1);
            imgb2.convertTo(imgb2, CV_8UC1);
            imgr2.convertTo(imgr2, CV_8UC1);
            cv::Mat img2(img.size(), CV_8UC3);
            vector<cv::Mat> channels;
            channels.push_back(imgb2);
            channels.push_back(imgg2);
            channels.push_back(imgr2);
            cv::merge(channels, img2);
            cv::imshow("img", img2); },
        &img);
    cv::createTrackbar(
        "Fix I", "img", 0, 200, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat imgr, imgg, imgb;
            Cimage_proc::color2rgb(img, imgr, imgg, imgb);
            // imshow("img",img);
            cv::imshow("imgr", imgr);
            cv::imshow("imgg", imgg);
            cv::imshow("imgb", imgb);
            cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
            imgr.convertTo(imgr, CV_64FC1);
            imgg.convertTo(imgg, CV_64FC1);
            imgb.convertTo(imgb, CV_64FC1);
            Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
            cv::Mat temp(img.size(), CV_64F, cv::Scalar(pos / 200.0));
            imgi=temp;
            cv::imshow("imgh", imgh);
            cv::imshow("imgs", imgs);
            cv::imshow("imgi", imgi);
            cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
            Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
            imgr2.convertTo(imgr2, CV_8UC1);
            imgg2.convertTo(imgg2, CV_8UC1);
            imgb2.convertTo(imgb2, CV_8UC1);
            cv::Mat img2(img.size(), CV_8UC3);
            vector<cv::Mat> channels;
            channels.push_back(imgb2);
            channels.push_back(imgg2);
            channels.push_back(imgr2);
            cv::merge(channels, img2);
            cv::imshow("img", img2); },
        &img);

    waitKey(0);
#elif ITEM == 2
    Mat img = imread(IMG_PATH);
    namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar(
        "Gamma Trans for H", "img", 0, 200, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat imgr, imgg, imgb;
            Cimage_proc::color2rgb(img, imgr, imgg, imgb);
            // imshow("img",img);
            cv::imshow("imgr", imgr);
            cv::imshow("imgg", imgg);
            cv::imshow("imgb", imgb);
            cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
            imgr.convertTo(imgr, CV_64FC1);
            imgg.convertTo(imgg, CV_64FC1);
            imgb.convertTo(imgb, CV_64FC1);
            Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
            Enhancement enh(imgh);
            imgh=enh.gammaTransform_f(pos/100.0);
            cv::imshow("imgh", imgh);
            cv::imshow("imgs", imgs);
            cv::imshow("imgi", imgi);
            cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
            Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
            imgr2.convertTo(imgr2, CV_8UC1);
            imgg2.convertTo(imgg2, CV_8UC1);
            imgb2.convertTo(imgb2, CV_8UC1);
            cv::Mat img2(img.size(), CV_8UC3);
            vector<cv::Mat> channels;
            channels.push_back(imgb2);
            channels.push_back(imgg2);
            channels.push_back(imgr2);
            cv::merge(channels, img2);
            cv::imshow("img", img2); },
        &img);
    cv::createTrackbar(
        "Gamma Trans for S", "img", 0, 200, [](int pos, void *userdata)
        {
            // cout<<log2(2)<<endl;
            cv::Mat img = *(Mat *)userdata;
            cv::Mat imgr, imgg, imgb;
            Cimage_proc::color2rgb(img, imgr, imgg, imgb);
            // imshow("img",img);
            cv::imshow("imgr", imgr);
            cv::imshow("imgg", imgg);
            cv::imshow("imgb", imgb);
            cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
            imgr.convertTo(imgr, CV_64FC1);
            imgg.convertTo(imgg, CV_64FC1);
            imgb.convertTo(imgb, CV_64FC1);
            Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
            Enhancement enh(imgs);
            imgs=enh.gammaTransform_f(pos/100.0);
            cv::imshow("imgh", imgh);
            cv::imshow("imgs", imgs);
            cv::imshow("imgi", imgi);
            cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
            Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
            imgr2.convertTo(imgr2, CV_8UC1);
            imgg2.convertTo(imgg2, CV_8UC1);
            imgb2.convertTo(imgb2, CV_8UC1);
            cv::Mat img2(img.size(), CV_8UC3);
            vector<cv::Mat> channels;
            channels.push_back(imgb2);
            channels.push_back(imgg2);
            channels.push_back(imgr2);
            cv::merge(channels, img2);
            cv::imshow("img", img2); },
        &img);
    cv::createTrackbar(
        "Gamma Trans for I", "img", 0, 200, [](int pos, void *userdata)
        {
          
            cv::Mat img = *(Mat *)userdata;
            cv::Mat imgr, imgg, imgb;
            Cimage_proc::color2rgb(img, imgr, imgg, imgb);
            cv::imshow("imgr", imgr);
            cv::imshow("imgg", imgg);
            cv::imshow("imgb", imgb);
            cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
            imgr.convertTo(imgr, CV_64FC1);
            imgg.convertTo(imgg, CV_64FC1);
            imgb.convertTo(imgb, CV_64FC1);
            Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
            Enhancement enh(imgi);
            imgi=enh.gammaTransform_f(pos/100.0);
            cv::imshow("imgh", imgh);
            cv::imshow("imgs", imgs);
            cv::imshow("imgi", imgi);
            cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
            Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
            imgr2.convertTo(imgr2, CV_8UC1);
            imgg2.convertTo(imgg2, CV_8UC1);
            imgb2.convertTo(imgb2, CV_8UC1);
            cv::Mat img2(img.size(), CV_8UC3);
            vector<cv::Mat> channels;
            channels.push_back(imgb2);
            channels.push_back(imgg2);
            channels.push_back(imgr2);
            cv::merge(channels, img2);
            cv::imshow("img", img2); },
        &img);
    waitKey(0);
#elif ITEM == 3
    cv::Mat img = imread(IMG_PATH);
    cv::Mat imgr, imgg, imgb;
    Cimage_proc::color2rgb(img, imgr, imgg, imgb);
    // imshow("img",img);
    cv::imshow("imgr", imgr);
    cv::imshow("imgg", imgg);
    cv::imshow("imgb", imgb);
    cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);

    imgr.convertTo(imgr, CV_64FC1);
    imgg.convertTo(imgg, CV_64FC1);
    imgb.convertTo(imgb, CV_64FC1);

    imgr /= 255;
    imgg /= 255;
    imgb /= 255;

    Filter filter_r(imgr);
    Filter filter_g(imgg);
    Filter filter_b(imgb);

    Mat imgr3 = filter_r.medianFilter(7);
    Mat imgg3 = filter_g.medianFilter(7);
    Mat imgb3 = filter_b.medianFilter(7);

    imshow("imgr3", imgr);
    imshow("imgb3", imgb);
    imshow("imgg3", imgg);

    Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
    Filter filterh(imgh);
    Filter filters(imgs);
    Filter filteri(imgi);
    cv::imshow("imgh", imgh);
    cv::imshow("imgs", imgs);
    cv::imshow("imgi", imgi);
    imgh = filterh.medianFilter(7);
    imgs = filters.medianFilter(7);
    imgi = filteri.medianFilter(7);
    cv::imshow("imgh2", imgh);
    cv::imshow("imgs2", imgs);
    cv::imshow("imgi2", imgi);
    cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
    Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
    imgr2.convertTo(imgr2, CV_8UC1);
    imgg2.convertTo(imgg2, CV_8UC1);
    imgb2.convertTo(imgb2, CV_8UC1);
    cv::Mat img2(img.size(), CV_8UC3);
    vector<cv::Mat> channels;
    channels.push_back(imgb2);
    channels.push_back(imgg2);
    channels.push_back(imgr2);
    cv::merge(channels, img2);
    cv::imshow("img", img2);
    vector<cv::Mat> channels3 = {imgb3, imgg3, imgr3};
    cv::Mat img3(img.size(), CV_64FC3);
    cv::merge(channels3, img3);
    cv::imshow("img3", img3);
    waitKey(0);

#elif ITEM == 4
    cv::Mat img = imread(IMG_PATH);
    cv::Mat imgr, imgg, imgb;
    Cimage_proc::color2rgb(img, imgr, imgg, imgb);
    cv::imshow("imgr", imgr);
    cv::imshow("imgg", imgg);
    cv::imshow("imgb", imgb);
    cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
    imgr.convertTo(imgr, CV_64FC1);
    imgg.convertTo(imgg, CV_64FC1);
    imgb.convertTo(imgb, CV_64FC1);
    imgr /= 255;
    imgg /= 255;
    imgb /= 255;
    Filter filter_r(imgr);
    Filter filter_g(imgg);
    Filter filter_b(imgb);
    Mat imgr3 = filter_r.laplace_enhence(3);
    Mat imgg3 = filter_g.laplace_enhence(3);
    Mat imgb3 = filter_b.laplace_enhence(3);
    imshow("imgr3", imgr);
    imshow("imgb3", imgb);
    imshow("imgg3", imgg);
    Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
    cv::imshow("imgh", imgh);
    cv::imshow("imgs", imgs);
    cv::imshow("imgi", imgi);
    Filter filterh(imgh);
    Filter filters(imgs);
    Filter filteri(imgi);

    imgh = filterh.laplace_enhence(3);
    imgs = filters.laplace_enhence(3);
    imgi = filteri.laplace_enhence(3);
    cv::imshow("imgh2", imgh);
    cv::imshow("imgs2", imgs);
    cv::imshow("imgi2", imgi);
    cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
    Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
    imgr2.convertTo(imgr2, CV_8UC1);
    imgg2.convertTo(imgg2, CV_8UC1);
    imgb2.convertTo(imgb2, CV_8UC1);
    cv::Mat img2(img.size(), CV_8UC3);
    vector<cv::Mat> channels;
    channels.push_back(imgb2);
    channels.push_back(imgg2);
    channels.push_back(imgr2);
    cv::merge(channels, img2);
    cv::imshow("img", img2);
    vector<cv::Mat> channels3 = {imgb3, imgg3, imgr3};
    cv::Mat img3(img.size(), CV_64FC3);
    cv::merge(channels3, img3);
    cv::imshow("img3", img3);
    waitKey(0);
#elif ITEM == 5
    cv::namedWindow("Trackbars");

    cv::createTrackbar("Low H", "Trackbars", &hMin, 255, updateTrackbarPos);
    cv::createTrackbar("High H", "Trackbars", &hMax, 255, updateTrackbarPos);
    cv::createTrackbar("Low S", "Trackbars", &sMin, 255, updateTrackbarPos);
    cv::createTrackbar("High S", "Trackbars", &sMax, 255, updateTrackbarPos);
    cv::createTrackbar("Low V", "Trackbars", &vMin, 255, updateTrackbarPos);
    cv::createTrackbar("High V", "Trackbars", &vMax, 255, updateTrackbarPos);

    cv::Mat img = imread(IMG_PATH3);
    while (true)
    {
        // Update the HSV range according to the current trackbar positions
        std::vector<std::vector<int>> hsv_range = {{hMin, sMin, vMin, hMax, sMax, vMax}};
        // Apply color_grab() with the current HSV range
        std::vector<cv::Mat> masks = Utils::color_grab(img, hsv_range);
        // Display the masks
        for (size_t i = 0; i < masks.size(); ++i)
        {
            cv::imshow("Mask " + std::to_string(i), masks[i]);
            // Convert mask to 3 channels
            cv::Mat mask3;
            cv::cvtColor(masks[i], mask3, cv::COLOR_GRAY2BGR);
            // Apply bitwise_and operation with the same type and same number of channels
            cv::Mat image;
            cv::bitwise_and(mask3, img, image);
            cv::imshow("image" + std::to_string(i), image);
            
        }
        // Break the loop if the user presses a key
        if (cv::waitKey(1) >= 0)
            break;
    }
#elif ITEM == 6
    cv::Mat img = cv::imread(IMG_PATH3);
    cv::namedWindow("image");
    cv::setMouseCallback("image", onMouse, 0);
    while (1)
    {
        cv::Mat img2 = img.clone();
        if (roi.width > 0 && roi.height > 0)
        {
            cv::rectangle(img2, roi, cv::Scalar(0, 255, 0), 1);
            // cv::meanStdDev(img(roi), mean, stddev);
            // cout<<"mean: "<<mean<<"std: "<<stddev<<endl;
            cv::Mat imgr, imgg, imgb;
            Cimage_proc::color2rgb(img, imgr, imgg, imgb);
            // imshow("img",img);
            cv::Mat imgh(img.size(), CV_64F), imgs(img.size(), CV_64F), imgi(img.size(), CV_64F);
            imgr.convertTo(imgr, CV_64FC1);
            imgg.convertTo(imgg, CV_64FC1);
            imgb.convertTo(imgb, CV_64FC1);
            imgr /= 255;
            imgg /= 255;
            imgb /= 255;
            Cimage_proc::rgb2hsi((double *)imgh.data, (double *)imgs.data, (double *)imgi.data, (double *)imgr.data, (double *)imgg.data, (double *)imgb.data, img.cols, img.rows);
            Mat img_hsi;
            cv::merge(vector<Mat>({imgh, imgs, imgi}), img_hsi);
            cv::Scalar mean, stddev;
            cv::meanStdDev(img_hsi(roi), mean, stddev);
            cout << "mean: " << mean * 255 << "std: " << stddev * 255 << endl;
            std::vector<std::vector<int>> hsi_range = {{4, 0, 66, 26, 140, 233}, {(int)((mean[0] - 3 * stddev[0]) * 255), (int)((mean[1] - 3 * stddev[1]) * 255), (int)((mean[2] - 3 * stddev[2]) * 255), (int)((mean[0] + 3 * stddev[0]) * 255), (int)((mean[1] + 3 * stddev[1]) * 255), (int)((mean[2] + 3 * stddev[2]) * 255)}};
            // std::vector<std::vector<int>> hsi_range = {{6,8,50,17,185,255},{(int)(mean[0] -2* stddev[0]) * 255, (int)(mean[1] - 2* stddev[1]) * 255, (int)(mean[2] -2*  stddev[2]) * 255, (int)(mean[0] +2*  stddev[0]) * 255, (int)(mean[1] + 2* stddev[1]) * 255, (int)(mean[2] + 2* stddev[2]) * 255}};
            // Apply color_grab() with the current HSV range
            std::vector<cv::Mat> masks = Utils::color_grab(img, hsi_range);

            for (size_t i = 0; i < masks.size(); ++i)
            {
                cv::imshow("Mask " + std::to_string(i), masks[i]);
                // Convert mask to 3 channels
                cv::Mat mask3;
                cv::cvtColor(masks[i], mask3, cv::COLOR_GRAY2BGR);
                // Apply bitwise_and operation with the same type and same number of channels
                cv::Mat image;
                cv::bitwise_and(mask3, img, image);
                cv::imshow("image_face" + std::to_string(i), image);
            }
            cv::Mat imgr2(img.size(), CV_64F), imgg2(img.size(), CV_64F), imgb2(img.size(), CV_64F);
            Cimage_proc::hsi2rgb((double *)imgr2.data, (double *)imgg2.data, (double *)imgb2.data, (double *)imgh.data, (double *)imgs.data, (double *)imgi.data, img.cols, img.rows);
        }
        cv::imshow("image", img2);
        if (cv::waitKey(20) == 27)
            break;
    }

#endif
    return 0;
}
