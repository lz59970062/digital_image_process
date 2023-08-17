#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"

using namespace cv;
using namespace std;

int main() {
    // 1.读取图片
    Mat img = imread("PeppersRGB.bmp" ,0);
    if (img.empty()) {
        cout << "Image not found" << endl;
        return -1;
    }
    // // 2.移动，水平，垂直翻转
    // Mat translated = img_transf(img, Transform(0, 1, 1, 50, 50), 0);
    // Mat flipped_horizontally = img_transf(img, Transform(0, -1, 1, img.cols, 0), 0);
    // Mat flipped_vertically = img_transf(img, Transform(0, 1, -1, 0, img.rows), 0);

    // // 3. 尺度缩放
    // Mat resized_1_5 = img_transf(img, Transform(0, 1.5, 1.5, 0, 0), 0);// 1.5倍 后向，双线性插值
    // Mat resized_3 = img_transf(img, Transform(0, 3, 3, 0, 0), 0);// 3倍 后向，双线性插值
    // Mat resized_5_forward = img_transf(img, Transform(0, 5, 5, 0, 0), 0);// 5倍 后向，双线性插值
    // Mat resized_5_backward = img_transf(img, Transform(0, 5, 5, 0, 0), 2);// 5倍 前向，双线性插值

    // // 4. Perform image rotation
    // double angle_30 = 30 * CV_PI / 180;
    // double angle_45 = 45 * CV_PI / 180;
    // Mat rotated_30 = img_transf(img, Transform(angle_30, 1, 1, 0, 0), 1);// 30度，后向，双线性插值
    // Mat rotated_45 = img_transf(img, Transform(angle_45, 1, 1, 0, 0), 1);// 45度，后向，双线性插值

    //6 缩小
    // Mat resized_0_8 = img_transf(img, Transform(0,0.8,0.8, 0, 0), 0);// 0.8倍 后向，双线性插值
    // Mat resized_0_5 = img_transf(img, Transform(0,0.5,0.5,0,0),0);//0.5倍 后向，双线性插值

    //7 比较不同插值算法 
    // Mat resized_1_5 = img_transf(img, Transform(0, 1.5, 1.5, 0, 0), 0);// 1.5倍 后向，双线性插值
    // Mat resized_1_5_1 = img_transf(img, Transform(0, 1.5, 1.5, 0, 0), 1);// 1.5倍 后向，最近邻插值
    // imshow("resized_1_5", resized_1_5);
    // imshow("resized_1_5_1", resized_1_5_1);
    // waitKey(0);

    // 5. Display images
    // imshow("Original Image", img);
    // imshow("Translated", translated);
    // imshow("Flipped Horizontally", flipped_horizontally);
    // imshow("Flipped Vertically", flipped_vertically);
    // imshow("Resized 1.5", resized_1_5);
    // imshow("Resized 3", resized_3);
    // imshow("Resized 5 Forward", resized_5_forward);
    // imshow("Resized 5 Backward", resized_5_backward);
    // imshow("Rotated 30", rotated_30);
    // imshow("Rotated 45", rotated_45);
    // imshow("Resized 0.8", resized_0_8);
    // imshow("Resized 0.5", resized_0_5);
    F_tr f(img);
    imshow("Original Image", img);
    imshow("fft_mag", f.mdft_c(0));
    imshow("fft_phase", f.mdft_phase());
    
    waitKey(0);
    return 0;
}
