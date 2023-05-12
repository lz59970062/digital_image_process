#include <iostream>
#include <opencv2/opencv.hpp>
#include "img_trans.h"
 

#define ITEM 2
#ifdef _WIN64||_WIN32
#define IMG_PATH "C:\\Users\\lz599\\Desktop\\digital_image_process\\ch5\\PeppersRGB.bmp"
#else
#define IMG_PATH "../PeppersRGB.bmp"
#endif
using namespace std; 
using namespace cv;

int main(int arvc, char *argv[])
{
     
#if ITEM == 1
    Mat img=imread(IMG_PATH,cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout<<"can not load image"<<endl;
        return -1;
    }
    imshow("raw" ,img);
    waitKey(0);
    Enhancement enh(img);
    std::vector<double> lin_a={0.5,0.8,1.2,1.5};
    std::vector<cv::Mat> lin_res;
    lin_res.reserve(4);
    for (auto i:lin_a)
    {
         
        cv::Mat tmp=enh.linearTransform(i,0);
        cv::putText(tmp,"a="+std::to_string(i),cv::Point(10,30),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,255,255));
        lin_res.push_back(tmp);
    }
    cv::hconcat(lin_res,enh.result_img);
    imshow("linear",enh.result_img);
    waitKey(0);
    std::vector<vector<double>> piece_a={{0,100,120,255},{0,100,120,200}};
    std::vector<vector<double>> piece_b={{0,150,160,255},{0,120,150,255}};
    std::vector<cv::Mat> piece_res;
    piece_res.reserve(2);
    for(int i=0;i<piece_a.size();i++){
        cv::Mat tmp =enh.piecewiseLinearTransform(piece_a[i],piece_b[i]);
        piece_res.push_back(tmp);
    }
    hconcat(piece_res,enh.result_img);
    imshow("piecewise",enh.result_img);
    waitKey(0);   
    cv::resize(img,img,cv::Size(img.rows/2,img.cols/2));
    // cv::imshow("img21",img);
    
    Enhancement enh2(img);
    std::vector<cv::Mat> gamma_res;
    for(double i=0.2;i<3;i+=0.2){
        cv::Mat tmp=enh2.gammaTransform(i);
        cv::putText(tmp,"gamma:"+std::to_string(i),cv::Point(20,20),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,255,255));
        gamma_res.push_back(tmp);
    }
    hconcat(gamma_res,enh2.result_img);
    imshow("gamma",enh2.result_img);
    waitKey(0);
    std::vector<cv::Mat> log_res;
    for(double i=0.5;i<1.5;i+=0.2){
        cv::Mat tmp=enh2.logTransform(i);
        cv::putText(tmp,"log:"+std::to_string(i),cv::Point(20,20),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,255,255));
        log_res.push_back(tmp);
    }
    hconcat(log_res,enh2.result_img);
    imshow("log",enh2.result_img);
    waitKey(0);

#else ITEM == 2
 
    Mat img = imread(IMG_PATH, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "can not load image" << endl;
        return -1;
    }
    imshow("img", img);
    waitKey(0);
    

#endif
    return 0;
}
