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
    Mat img_double(img.rows, img.cols, CV_64FC3);
    img.convertTo(img_double, CV_64FC3);
    vector<Mat> img20, img50, img100;
    Mat res20(img.size(), img.type(),Scalar(0,0,0)), res50(img.size(), img.type(), Scalar(0, 0, 0)), res100(img.size(), img.type(), Scalar(0, 0, 0));
    for (int i = 0; i < 20; i++) {
        img20.push_back(img);
    }
    for (int i = 0; i < 50; i++) {
        img50.push_back(img);
    }
    for (int i = 0; i < 100; i++) {
        img100.push_back(img);
    }
    cout << img20.size() << endl;
    cout << img50.size() << endl; 
    cout << img100.size() << endl;
    
    for (const auto  i : img20) {
        Noise img_guss(img, "guss");

        res20 += i;
    }
    res20 = res20 / 20-20;
    imshow("img", res20);
    // 输出vector占用的内存空间
  
    waitKey(0);
#elif ITEM==4
   for(int i=0;i<100;i++){
    cout<< generateGaussianNoise(10, 10)[0]<<endl;
   }
#endif
    return 0;

}
