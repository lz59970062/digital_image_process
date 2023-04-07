#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"

using namespace cv;
using namespace std;

int main() {
    // 1. Read image in grayscale
    Mat img = imread("fenjin.jpg");
    if (img.empty()) {
        cout << "Image not found" << endl;
        return -1;
    }

    // 2. Perform translation, horizontal and vertical flipping
    Mat translated = img_transf(img, Transform(0, 1, 1, 50, 50), 0);
    Mat flipped_horizontally = img_transf(img, Transform(0, -1, 1, img.cols, 0), 0);
    Mat flipped_vertically = img_transf(img, Transform(0, 1, -1, 0, img.rows), 0);

    // 3. Perform image resizing
    Mat resized_1_5 = img_transf(img, Transform(0, 1.5, 1.5, 0, 0), 0);
    Mat resized_3 = img_transf(img, Transform(0, 3, 3, 0, 0), 0);
    Mat resized_5_forward = img_transf(img, Transform(0, 5, 5, 0, 0), 0);
    Mat resized_5_backward = img_transf(img, Transform(0, 5, 5, 0, 0), 1);

    // 4. Perform image rotation
    double angle_30 = 30 * CV_PI / 180;
    double angle_45 = 45 * CV_PI / 180;
    Mat rotated_30 = img_transf(img, Transform(angle_30, 1, 1, 0, 0), 1);
    Mat rotated_45 = img_transf(img, Transform(angle_45, 1, 1, 0, 0), 1);

    // 5. Display images
    imshow("Original Image", img);
    imshow("Translated", translated);
    imshow("Flipped Horizontally", flipped_horizontally);
    imshow("Flipped Vertically", flipped_vertically);
    imshow("Resized 1.5", resized_1_5);
    imshow("Resized 3", resized_3);
    imshow("Resized 5 Forward", resized_5_forward);
    imshow("Resized 5 Backward", resized_5_backward);
    imshow("Rotated 30", rotated_30);
    imshow("Rotated 45", rotated_45);

    waitKey(0);
    return 0;
}
