#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "myfunc.h"

using namespace std;
using namespace cv;

const double PI = 3.141592653589793;

Mat wh = (Mat_<double>(2, 1) << 256, 256);
int main()
{
    Mat img(wh.at<double>(1), wh.at<double>(0), CV_8UC3, Scalar(0, 0, 255));
    for (int i = 0; i < img.rows; i++)
    {
        auto ptr = img.ptr<Vec3b>(i);
        for (int j = 0; j < img.cols; j++)
        {
            ptr[j] = Vec3b(i, j, i);
        }
    }
    float theta = 0;
    while (1)
    {

        theta += 0.001;
        Mat T=Transform(theta, 3,4, 0, 0);
 
        Mat img2 = img_transf(img, T, 1);
  
        imshow("img", img2);
        auto k = waitKey(4);
        if (k == 27)
            break;
    }
    return 0;

}
