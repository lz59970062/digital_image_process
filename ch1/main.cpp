#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

const double PI = 3.141592653589793;


Mat wh = (Mat_<double>(2, 1) << 256, 256);
int main() {
    Mat img(wh.at<double>(1), wh.at<double>(0), CV_8UC3, Scalar(0, 0, 255));
    for (int i = 0; i < img.rows; i++) {
        auto ptr = img.ptr<Vec3b>(i);
        for (int j = 0; j < img.cols; j++) {
            ptr[j] = Vec3b(i, j, i);
        }
    }
    
    for (double theta = 0; theta < 2 * PI; theta += 0.001) {
        Mat rot_mat = (Mat_<double>(3, 3) << cos(theta), -sin(theta), 0, sin(theta), cos(theta), 0, 0, 0, 1);
        Mat whrot = (Mat_<double>(2, 2) << sin(theta), cos(theta), cos(theta), sin(theta));
        Mat wh_new = whrot * wh;
        Mat rot_matinv=rot_mat.inv();
        if (wh_new.at<double>(0) < 10) continue;
        Mat img2(wh_new.at<double>(1), wh_new.at<double>(0), CV_8UC3, Scalar(0, 0, 0));
        for (auto i = 0; i < img2.rows; i++) {
            auto ptr = img2.ptr<Vec3b>(i);
            for (auto j = 0; j < img2.cols; j++) {
                Mat p = (Mat_<double>(3, 1) << j, i, 1);
                Mat p2 = rot_matinv * p;
                if (p2.at<double>(0, 0) >= 0 && p2.at<double>(0, 0) < img.cols && p2.at<double>(1, 0) >= 0 && p2.at<double>(1, 0) < img.rows) {
                    ptr[j] = img.at<Vec3b>(p2.at<double>(1, 0), p2.at<double>(0, 0));
                }
            }
        }
        //imshow("img1", img);
        imshow("Image", img2);
        waitKey(1);
    }
    return 0;
}
