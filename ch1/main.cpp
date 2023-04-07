// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <cmath>
// #include "myfunc.h"

// using namespace std;
// using namespace cv;

// const double PI = 3.141592653589793;

// Mat wh = (Mat_<double>(2, 1) << 256, 256);
// int main()
// {
//     Mat img1 = imread("fenjin.jpg",0);
//     Mat thresh;
//      adaptiveThreshold(img1, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

//     // 显示结果
//     namedWindow("adaptive threshold", WINDOW_NORMAL);
//     imshow("adaptive threshold", thresh);
//     waitKey(0);
//     destroyAllWindows();
//     Mat img = imread("fenjin.jpg");
//     img_inverse(img);
//     imshow("img", img);
//     waitKey(0);

//     for (float i = 0; i < 0.8; i += 0.2)
//     {
//         img = imread("fenjin.jpg");
//         saltandpepper(img, 0.1);
//         imshow("img", img);
//         waitKey(0);
//     }
//     for (int i = 0; i < 240; i += 20)
//     {
//         img = imread("fenjin.jpg",0);
//         img_binary(img, i);
//         imshow("img", img);
//         waitKey(0);
//     }
// }
// // Mat img(wh.at<double>(1), wh.at<double>(0), CV_8UC3, Scalar(0, 0, 255));
// // for (int i = 0; i < img.rows; i++)
// // {
// //     auto ptr = img.ptr<Vec3b>(i);
// //     for (int j = 0; j < img.cols; j++)
// //     {
// //         ptr[j] = Vec3b(i, j, i);
// //     }
// // }

// // float theta = 0;
// // while (1)
// // {
// //     theta += 0.1;
// //     Mat T=Transform(theta, 3,3, 0, 0);

// //     Mat img2 = img_transf(img, T, 1);

// //     imshow("img", img2);
// //     auto k = waitKey(4);
// //     if (k == 27)
// //         break;
// // }
// // return 0;

#include <opencv2/opencv.hpp>
#include <iostream>
#include "myfunc.h"

using namespace cv;

int main()
{
    // 读入图像
    Mat img = imread("fenjin.jpg", IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Failed to read image!" << std::endl;
        return -1;
    }

    // 图像二值化
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    Mat thresh = img_binary(gray_img, 128);

    // 图像反色
    Mat invert = img_inverse(img.clone());

    // 添加椒盐噪声
    Mat noisy_img = img.clone();
    saltandpepper(noisy_img, 0.1);

    // 显示结果
    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Binary Threshold Image", WINDOW_NORMAL);
    namedWindow("Inverted Image", WINDOW_NORMAL);
    namedWindow("Noisy Image", WINDOW_NORMAL);

    imshow("Original Image", img);
    imshow("Binary Threshold Image", thresh);
    imshow("Inverted Image", invert);
    imshow("Noisy Image", noisy_img);

    waitKey(0);
    destroyAllWindows();

    return 0;
}


