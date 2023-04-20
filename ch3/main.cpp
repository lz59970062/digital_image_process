#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"

#define ITEM 4
int main(int arvc, char *argv[])
{

    Mat img;
#if ITEM == 1
    for (float i = 0; i < 0.8; i += 0.2)
    {
        img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\ch3\\PeppersRGB.bmp");
        saltandpepper(img, i);
        imshow("img", img);
        waitKey(0);
    }
#elif ITEM == 2
    img = imread("PeppersRGB.bmp");
    Noise noise(img, "Gaussian");
    Mat img2 = noise.gaussian(0, 50);
    hconcat(img, img2, img);
    imshow("img", img);

    waitKey(0);

#elif ITEM == 3
    img = imread("PeppersRGB.bmp");
    Mat img_double(img.rows, img.cols, CV_64FC3);
    img.convertTo(img_double, CV_64FC3);
    Mat imgSum(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Noise noise(img, "Gaussian");
    for (int i = 0; i < 10; i++)
    {
        
        Mat img2 = noise.gaussian(5, 80);
        img2.convertTo(img2, CV_64FC3);
        imgSum = imgSum + img2;
    }
    Mat imgMean = imgSum / 10;
    imgMean.convertTo(imgMean, CV_8UC3);
    hconcat(img, imgMean, img);
    hconcat(img,noise.gaussian(5,80),img);
    imshow("img", img);
    
    waitKey(0);
#elif ITEM==4
   for(int i=0;i<100;i++){
    cout<< generateGaussianNoise(10, 10)[0]<<endl;
   }
#endif
    return 0;

}
