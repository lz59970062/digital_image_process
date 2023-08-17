#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

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
                line(histImage_b, Point(bin_w * (i - 1), hist_h - cvRound(hist[0].at<float>(i - 1))),
                     Point(bin_w * (i), hist_h - cvRound(hist[0].at<float>(i))),
                     Scalar(255, 0, 0), 2, 8, 0);
                line(histImage_g, Point(bin_w * (i - 1), hist_h - cvRound(hist[1].at<float>(i - 1))),
                     Point(bin_w * (i), hist_h - cvRound(hist[1].at<float>(i))),
                     Scalar(0, 255, 0), 2, 8, 0);
                line(histImage_r, Point(bin_w * (i - 1), hist_h - cvRound(hist[2].at<float>(i - 1))),
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

class Noise
{
public:
    Noise(Mat &img, string name)
        : img(img.clone()), name(name) {}

    Mat saltandpepper(double f);
    Mat gaussian(float mean, float sigma);
    Mat normal(uint8_t a, uint8_t b);

private:
    string name;
    Mat img;
    Mat noise_rate;

    vector<Vec3b> generateGaussianNoise(double mu, double sigma);
};

Mat Noise::saltandpepper(double f)
{
    Mat image = this->img.clone();
    int n1 = static_cast<int>(image.rows * image.cols * f / 2);
    int n2 = static_cast<int>(image.rows * image.cols * f / 2);
    for (int k = 0; k < n1; k++)
    {
        int i = rand() % image.cols;
        int j = rand() % image.rows;
        if (image.channels() == 1)
        {
            image.at<uchar>(j, i) = 255;
        }
        else if (image.channels() == 3)
        {

            image.at<cv::Vec3b>(j, i)[0] = 255;
            image.at<cv::Vec3b>(j, i)[1] = 255;
            image.at<cv::Vec3b>(j, i)[2] = 255;
        }
    }
    for (int k = 0; k < n2; k++)
    {
        int i = rand() % image.cols;
        int j = rand() % image.rows;
        if (image.channels() == 1)
        {
            image.at<uchar>(j, i) = 0;
        }
        else if (image.channels() == 3)
        {
            image.at<cv::Vec3b>(j, i)[0] = 0;
            image.at<cv::Vec3b>(j, i)[1] = 0;
            image.at<cv::Vec3b>(j, i)[2] = 0;
        }
    }

    return image;
}

vector<Vec3b> Noise::generateGaussianNoise(double mu, double sigma)
{
    const double epsilon = numeric_limits<double>::min();
    double z0, z1, u1, u2;
    double data[6];

    for (int i = 0; i < 3; i++)
    {
        do
        {
            u1 = rand() * (1.0 / RAND_MAX);
            u2 = rand() * (1.0 / RAND_MAX);
        } while (u1 <= epsilon);

        z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
        z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
        data[2 * i] = z0 * sigma + mu;
        data[2 * i + 1] = z1 * sigma + mu;
    }
    vector<Vec3b> noise;
    noise.push_back(Vec3b(data[0], data[1], data[2]));
    noise.push_back(Vec3b(data[3], data[4], data[5]));
    return noise;
}
Mat Noise::gaussian(float mean, float sigma)
{
    Mat image = this->img.clone();
    int channels = image.channels();
    int cols = image.cols;
    int rows = image.rows;
    mean+= sigma;
    if (channels == 1) {
        cv::cvtColor(image,image, cv::COLOR_GRAY2BGR);
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j += 2)
        {
            vector<Vec3b> noise = generateGaussianNoise(mean, sigma);
            image.at<Vec3b>(i, j) = image.at<Vec3b>(i, j) + noise[0];
            image.at<Vec3b>(i, j + 1) = image.at<Vec3b>(i, j + 1) + noise[1];
        }
    }
    if (channels == 1) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    image-= sigma;
    return image;
}
Mat Noise::normal(uint8_t a, uint8_t b)
{
    Mat img = this->img.clone();
    int rows = img.rows;
    int cols = img.cols;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (img.channels() == 3)
            {
                img.at<Vec3b>(i, j) = img.at<Vec3b>(i, j) + Vec3b((rand() + a) % b, (rand() + a) % b, (rand() + a) % b);
            }
            else
            {
                img.at<uchar>(i, j) = img.at<uchar>(i, j) + (rand() + a) % b;
            }
        }
    }
    return img;
}


// class HistogramEqualizer : public Histogram
// {
// public:
//     Mat equalize(const Mat &image)
//     {
//         Mat equalizedImage;
//         if (image.channels() == 1)
//         {
//             // 对于灰度图像
//             equalizeHist(image, equalizedImage);
//         }
//         else
//         {
//             // 对于彩色图像
//             vector<Mat> bgr_planes;
//             split(image, bgr_planes);
//             for (int i = 0; i < image.channels(); ++i)
//             {
//                 equalizeHist(bgr_planes[i], bgr_planes[i]);
//             }
//             merge(bgr_planes, equalizedImage);
//         }
//         return equalizedImage;
//     }

//     vector<int> computeHistogramInt(const Mat& image) {
//         vector<Mat> hist = computeHistogram(image);
//         vector<int> histInt;
//         histInt.reserve(histSize);

//         for(int i = 0; i < histSize; ++i) {
//             histInt.push_back(cvRound(hist.at<float>(i)));
//         }
//         return histInt;
//     }

//     // 直方图匹配（规定化）
//     Mat matchHistogram(const Mat &image, const Mat &reference)
//     {
//         Mat matchedImage;
//         // 暂时只对灰度图像进行处理
//         if (image.channels() == 1 && reference.channels() == 1)
//         {
//             Mat lut(1, 256, CV_8U);
//             vector<int> hist = Histogram::computeHistogramInt(image);
//             vector<int> refHist = Histogram::computeHistogramInt(reference);

//             vector<float> probHist(256, 0), probRefHist(256, 0);
//             for (int i = 0; i < 256; ++i)
//             {
//                 probHist[i] = (float)hist[i] / (image.cols * image.rows);
//                 probRefHist[i] = (float)refHist[i] / (reference.cols * reference.rows);
//             }

//             vector<float> cdfHist(256, 0), cdfRefHist(256, 0);
//             cdfHist[0] = probHist[0];
//             cdfRefHist[0] = probRefHist[0];
//             for (int i = 1; i < 256; ++i)
//             {
//                 cdfHist[i] = cdfHist[i - 1] + probHist[i];
//                 cdfRefHist[i] = cdfRefHist[i - 1] + probRefHist[i];
//             }

//             for (int i = 0; i < 256; ++i)
//             {
//                 int j = 255;
//                 do
//                 {
//                     lut.at<uchar>(i) = j;
//                     j--;
//                 } while (j >= 0 && cdfHist[i] <= cdfRefHist[j]);
//             }

//             LUT(image, lut, matchedImage);
//         }
//         return matchedImage;
//     }
// };