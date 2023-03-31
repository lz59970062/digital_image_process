
#include "myfunc.h"
using namespace cv;
void slatNosise(cv::Mat &img, double f)
{
    int i, j;
    int n = img.rows;
    int m = img.cols;
    for (i = 0; i < img.rows; i++)
    {
        for (j = 0; j < img.cols; j++)
        {
            // select channel
            if (img.channels() > 1)
            {
                img.at<cv::Vec3b>(i, j)[0] = img.at<cv::Vec3b>(i, j)[0] + f * (rand() % 256 - 128);
                img.at<cv::Vec3b>(i, j)[1] = img.at<cv::Vec3b>(i, j)[1] + f * (rand() % 256 - 128);
                img.at<cv::Vec3b>(i, j)[2] = img.at<cv::Vec3b>(i, j)[2] + f * (rand() % 256 - 128);
            }
            else
            {
                img.at<uchar>(i, j) = img.at<uchar>(i, j) + f * (rand() % 256 - 128);
            }
        }
    }
}

// binary
Mat &img_binary(Mat &img, uint8_t thr)
{
    if (img.empty())
    {
        std::cout << "img is empty" << std::endl;
        return img;
    }
    if (img.type()!=CV_8UC1)
    {
        std::cout << "img type is not CV_8UC1,unsupport opt" << std::endl;
        return img;
    }
    for (int i=0;i<img.rows;i++){
        for (int j=0;j<img.cols;j++){
            if (img.at<uchar>(i,j)>thr){
                img.at<uchar>(i,j)=255;
            }else{
                img.at<uchar>(i,j)=0;
            }
        }
    }
    return img;
}


Mat &img_inverse(Mat &img){
    if(img.empty()){
        std::cout<<"img is empty"<<std::endl;
        return img;
    }
}