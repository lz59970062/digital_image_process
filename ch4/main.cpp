#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"
#include "histogram.hpp"
#define ITEM 1
int main(int arvc, char *argv[])
{

    Mat img;
    // cout<<"ok   "<<e
    img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\ch4\\PeppersRGB.bmp", cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    cv::imshow("img",img);

#if ITEM == 1
     

#endif
    return 0;
}
