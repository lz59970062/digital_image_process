#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"

#define ITEM 3
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
    img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\ch3\\PeppersRGB.bmp");
    Mat imgd(img.size(), CV_64FC3);
    img.convertTo(imgd, CV_64FC3);
    Mat imgSum(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Mat res10(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Mat res20(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Mat res50(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Noise noise(img, "Gaussian");
    for (int i = 0; i < 50; i++)
    {
        Mat img2 = noise.gaussian(80, 80);
        img2.convertTo(img2, CV_64FC3);
        imgSum = imgSum + img2; 
        if(i==9)
        {
            res10=imgSum/10-80;
        }
        else if(i==19)
        {
            res20=imgSum/20-80;
        }
        else if(i==49)
        {
            res50=imgSum/50-80;
        }
    }
    vector<Mat> Res;
    res10.convertTo(res10, CV_8UC3);
    res20.convertTo(res20, CV_8UC3);
    res50.convertTo(res50, CV_8UC3);
    putText(img, "Raw", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    Res.emplace_back(img);
    putText(res10, "10", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    Res.emplace_back(res10);
    putText(res20, "20", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    Res.emplace_back(res20);
    putText(res50, "50", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    Res.emplace_back(res50);
    Mat img2;
    hconcat(Res, img2);
    imshow("img",img2);

    waitKey(0);
#elif ITEM == 4
    Vec3f sum=Vec3f(0,0,0);
    for (int i = 0; i < 1000; i++)
    {
        sum+=(generateGaussianNoise(10, 10)[0]);
        cout << sum << endl;

    }
    cout<<"avg"<<sum/1000<<endl;
#endif
    return 0;
}
