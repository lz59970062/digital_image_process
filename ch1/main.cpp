#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int main(){
    Mat img(480,640,CV_8UC3,Scalar(0,0,255));
    imshow("Image",img);
}