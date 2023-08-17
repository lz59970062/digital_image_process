#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "img_proc.hpp"
using namespace std;
using namespace cv;

class Histogram
{
private:
    int histSize;
    float range[2];
    const float *histRange;
    bool uniform;
    bool accumulate;

public:
    Histogram() : histSize(256), range{0.0, 256.0}, histRange{range}, uniform(true), accumulate(false) {}
    std::vector<Mat> computeHistogram(const Mat &image)
    {
        Mat hist;
        vector<Mat> result_planes;
        if (image.channels() == 1)
        {
            // 对于灰度图像
            result_planes.reserve(1);
            calcHist(&image, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
            result_planes.push_back(hist);
        }
        else
        {
            // 对于彩色图像
            vector<Mat> bgr_planes;
            bgr_planes.reserve(3);
            result_planes.reserve(3);
            split(image, bgr_planes);
            for (int i = 0; i < image.channels(); ++i)
            {
                calcHist(&bgr_planes[i], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
                result_planes.push_back(hist);
            }
        }
        return result_planes;
    }
    void drawHistogram(const vector<Mat> &hist, const string &windowName)
    {
        // 创建直方图画布
        int hist_w = 512, hist_h = 400;
        int bin_w = cvRound((double)hist_w / histSize);
        vector<Mat> res;
        if (hist.size() == 1)
        {
            Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
            // 归一化直方图以适应画布
            normalize(hist[0], hist[0], 0, histImage.rows, NORM_MINMAX, -1, Mat());
            for (int i = 1; i < histSize; ++i)
            {
                line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist[0].at<float>(i))),
                     Point(bin_w * (i), hist_h),
                     Scalar(255, 255, 255), 2, 8, 0);
            }
            res.push_back(histImage);
            hconcat(res, histImage);
            namedWindow(windowName, WINDOW_AUTOSIZE);
            imshow(windowName, histImage);
        }
        else
        {
            Mat histImage_b(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
            Mat histImage_g(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
            Mat histImage_r(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
            // 归一化直方图以适应画布
            normalize(hist[0], hist[0], 0, histImage_b.rows, NORM_MINMAX, -1, Mat());
            normalize(hist[1], hist[1], 0, histImage_g.rows, NORM_MINMAX, -1, Mat());
            normalize(hist[2], hist[2], 0, histImage_r.rows, NORM_MINMAX, -1, Mat());
            // 对于彩色图像
            for (int i = 1; i < histSize; ++i)
            {
                line(histImage_b, Point(bin_w * (i - 1), hist_h),
                     Point(bin_w * (i), hist_h - cvRound(hist[0].at<float>(i))),
                     Scalar(255, 0, 0), 2, 8, 0);
                line(histImage_g, Point(bin_w * (i - 1), hist_h),
                     Point(bin_w * (i), hist_h - cvRound(hist[1].at<float>(i))),
                     Scalar(0, 255, 0), 2, 8, 0);
                line(histImage_r, Point(bin_w * (i - 1), hist_h),
                     Point(bin_w * (i), hist_h - cvRound(hist[2].at<float>(i))),
                     Scalar(0, 0, 255), 2, 8, 0);
            }
            res.push_back(histImage_b);
            res.push_back(histImage_g);
            res.push_back(histImage_r);
            Mat histImage;
            hconcat(res, histImage);
            namedWindow(windowName, WINDOW_AUTOSIZE);
            imshow(windowName, histImage);
            // 显示直方图
        }
    }
};

class HistogramEqualizer : public Histogram
{
public:
    Mat equalize(const Mat &image)
    {
        Mat equalizedImage;
        if (image.channels() == 1)
        {
            // 对于灰度图像
            equalizeHist(image, equalizedImage);
        }
        else
        {
            // 对于彩色图像
            vector<Mat> bgr_planes;
            split(image, bgr_planes);
            for (int i = 0; i < image.channels(); ++i)
            {
                equalizeHist(bgr_planes[i], bgr_planes[i]);
            }
            merge(bgr_planes, equalizedImage);
        }
        return equalizedImage;
    }

    // // 直方图匹配（规定化）
    // Mat matchHistogram(const Mat &image, const Mat &reference)
    // {
    //     Mat matchedImage;
    //     // 暂时只对灰度图像进行处理
    //     if (image.channels() == 1 && reference.channels() == 1)
    //     {
    //         Mat lut(1, 256, CV_8U);
    //         vector<int> hist = Histogram::computeHistogramInt(image);
    //         vector<int> refHist = Histogram::computeHistogramInt(reference);

    //         vector<float> probHist(256, 0), probRefHist(256, 0);
    //         for (int i = 0; i < 256; ++i)
    //         {
    //             probHist[i] = (float)hist[i] / (image.cols * image.rows);
    //             probRefHist[i] = (float)refHist[i] / (reference.cols * reference.rows);
    //         }

    //         vector<float> cdfHist(256, 0), cdfRefHist(256, 0);
    //         cdfHist[0] = probHist[0];
    //         cdfRefHist[0] = probRefHist[0];
    //         for (int i = 0; i < 255; ++i)
    //         {
    //             cdfHist[i] = cdfHist[i ] + probHist[i];
    //             cdfRefHist[i] = cdfRefHist[i ] + probRefHist[i];
    //         }

    //         for (int i = 0; i < 256; ++i)
    //         {
    //             int j = 255;
    //             do
    //             {
    //                 lut.at<uchar>(i) = j;
    //                 j--;
    //             } while (j >= 0 && cdfHist[i] <= cdfRefHist[j]);
    //         }

    //         LUT(image, lut, matchedImage);
    //     }
    //     return matchedImage;
    // }
};

class Utils
{
public:
    static vector<cv::Mat> color_grab(  cv::Mat &img, vector<vector<int>> hsv_range)
    {
      
        // cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        vector<cv::Mat> Masks;
        vector<vector<double>> hsv_rangef(hsv_range.size(), vector<double>(hsv_range[0].size()));
        for (size_t i = 0; i < hsv_range.size(); ++i)
        {
            for (size_t j = 0; j < hsv_range[i].size(); ++j)
            {
                auto temp=static_cast<double>(hsv_range[i][j]) / 255.0;
                if (temp<0) temp=0;
                if (temp>255) temp=255;
                hsv_rangef[i][j] = temp ;
            }
        }
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
        for (auto &i : hsv_rangef) // Change to hsv_rangef
        {
            cv::Mat mask;
            cv::inRange(img_hsi, cv::Scalar(i[0], i[1], i[2]), cv::Scalar(i[3], i[4], i[5]), mask);
            // opening = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)  # 对得到的mask进行开运算
            // cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
            Masks.push_back(mask);
        }
        return Masks;
    }
};
