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
    img = imread("./PeppersRGB.bmp");
    if (img.empty())
        cout << "empty image" << endl;
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
        if (i == 9)
        {
            res10 = imgSum / 10 - 50;
        }
        else if (i == 19)
        {
            res20 = imgSum / 20 - 50;
        }
        else if (i == 49)
        {
            res50 = imgSum / 50 - 50;
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
    imshow("img", img2);
    imwrite("result2.jpg", img2);
    waitKey(0);
#elif ITEM == 4
    img = imread("./PeppersRGB.bmp");
    if (img.empty())
        cout << "empty image" << endl;
    Mat imgd(img.size(), CV_64FC3);
    img.convertTo(imgd, CV_64FC3);
    Mat imgSum1(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Mat imgSum2(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Mat resguss(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Mat ressalt(img.size(), CV_64FC3, Scalar(0, 0, 0));
    Noise guss(img, "guss");
    Noise salt(img, "salt");
    Mat guss_noise;
    Mat salt_noise;
    for (int i = 0; i < 100; i++)
    {
        guss_noise = guss.gaussian(50, 50);
        salt_noise = salt.saltandpepper(0.1);
        guss_noise.convertTo(guss_noise, imgSum1.type());
        salt_noise.convertTo(salt_noise, imgSum2.type());
        imgSum1 += guss_noise;
        imgSum2 += salt_noise;
    }
    imgSum1 /= 100;
    imgSum1 -= 50;
    imgSum2 /= 100;
    imgSum1.convertTo(imgSum1, CV_8UC3);
    imgSum2.convertTo(imgSum2, CV_8UC3);

    vector<Mat> Res;
    putText(img, "Raw", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    Res.emplace_back(img);
    putText(imgSum1, "guss", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    Res.emplace_back(imgSum1);
    putText(imgSum2, "salt", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    Res.emplace_back(imgSum2);
    Mat img2;
    hconcat(Res, img2);
    imshow("img", img2);
    imwrite("result3.jpg", img2);
    vector<Mat> Res2;
    guss_noise.convertTo(guss_noise, CV_8UC3);
    salt_noise.convertTo(salt_noise, CV_8UC3);
    Res2.emplace_back(guss_noise);
    Res2.emplace_back(salt_noise);
    hconcat(Res2, img2);
    imwrite("result4.jpg", img2);
    imshow("img wiit Noise", img2);
    waitKey(0);
#elif ITEM == 5
    img = imread("PeppersRGB.bmp");
    vector<Mat> Res;
    Mat T =Transform(0,1,1,20,20);
    Mat img2=img_transf(img,T,0);
    Mat img3=img2-img;
    Res.emplace_back(img3);
    Mat img4=img2/img;
    Res.emplace_back(img4);
    Mat img5=img2+img;
    Res.emplace_back(img5);
    Res.emplace_back(img);
    Res.emplace_back(img2);
    hconcat(Res,img5);
    imshow("img",img5);
    imwrite("result5.jpg",img5);
    waitKey(0);

#endif
    return 0;
}
