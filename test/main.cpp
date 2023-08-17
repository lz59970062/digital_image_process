#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"
using namespace cv;
using namespace std;
#define PI 3.141592653
int main() {
    // 1.读取图片
    Mat img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\test\\fenjin.jpg" );
    //cv::resize(img, img, cv::Size(img.rows / 2, img.cols / 2), 0, 0, INTER_LINEAR);
    if (img.empty()) {
        cout << "Image not found" << endl;
        return -1;
    }
    cv::imshow("raw img", img);
    Mat resized_1 = img_transf(img, Transform(PI / 6, 1, 1, 0, 0), 2);
    Mat resized_2 = img_transf(img, Transform(PI/6 , 1,1, 0, 0),1);
    Mat resized_3 = img_transf(img, Transform(PI/6, 1, 1, 0,0), 0);
    imshow("resized_1 cubic", resized_1);
    imshow("resized_2 nearst", resized_2);
    imshow("resized_3 b Liner", resized_3);
    waitKey(0);
    return 0;
}
