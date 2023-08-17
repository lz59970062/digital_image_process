#include "img_trans.h"
#include <vector>
#include <iostream>
 

Enhancement::Enhancement(const cv::Mat &img)
{
    if( img.empty() )
    {
        std::cerr<<"Image is empty!"<<std::endl;
        return;
    }
    if(img.channels()==3){
        cv::cvtColor(img,raw_image,cv::COLOR_BGR2RGB);
    }
    else
    raw_image = img.clone();

}
Enhancement::~Enhancement()
{
    std::cout<<"Enhancement class destructed"<<std::endl;
}

cv::Mat Enhancement::linearTransform(const double &a, const double &b)  
{
    this->result_img=this->raw_image.clone();
    for (int i=0;i<this->raw_image.rows;i++)
    {
        for (int j=0;j<this->raw_image.cols;j++){
            this->result_img.at<uchar>(i,j)=cv::saturate_cast<uchar>(a*this->raw_image.at<uchar>(i,j)+b);
        }
    }
    return this->result_img;
}
cv::Mat Enhancement::piecewiseLinearTransform(const std::vector<double> &a, const std::vector<double> &b){
    
    for(int i=0;i<a.size();i++){
        if(a[i]<0||a[i]>255||b[i]<0||b[i]>255){
            std::cerr<<" out of range"<<std::endl;
            return this->raw_image;
        }
        if(( a[i]<=a[i-1])||( b[i]<=b[i-1])){
            std::cerr<<"a is not increasing"<<std::endl;
            return this->raw_image;
        }
    }
    this->result_img=this->raw_image.clone();
    int piece_num=a.size()+1;
    for (int i=0;i<this->raw_image.rows;i++)
    {
        for (int j=0;j<this->raw_image.cols;j++){
            int index=0;
            while (index<piece_num-1)
            {
                if (this->raw_image.at<uchar>(i,j)>=a[index]&&this->raw_image.at<uchar>(i,j)<a[index+1])
                {
                    result_img.at<uchar>(i,j)=cv::saturate_cast<uchar>(b[index+1]-b[index])/(a[index+1]-a[index])*(this->raw_image.at<uchar>(i,j)-a[index])+b[index];
                    break;
                }
                index++;
            }
            // if (index==piece_num-1)
            // {
            //     result_img.at<uchar>(i,j)=cv::saturate_cast<uchar>(b[index]);
            // }
        }
    }
    return this->result_img;
}

cv::Mat Enhancement::gammaTransform(const double &gamma){
    this->result_img=this->raw_image.clone();
    this->result_img.convertTo(this->result_img,CV_64F);
    for (int i=0;i<this->raw_image.rows;i++)
    {
        for (int j=0;j<this->raw_image.cols;j++){
            this->result_img.at<double >(i,j)= pow(this->raw_image.at<uchar>(i,j)/255.0,gamma) ;
        }
    }
    this->result_img*=255;
    this->result_img.convertTo(this->result_img,CV_8U);
    return this->result_img;
}
cv::Mat Enhancement::gammaTransform_f(const double &gamma){
    this->result_img=this->raw_image.clone();
    this->result_img.convertTo(this->result_img,CV_64F);
    for (int i=0;i<this->raw_image.rows;i++)
    {
        for (int j=0;j<this->raw_image.cols;j++){
            this->result_img.at<double >(i,j)= pow(this->raw_image.at<double>(i,j),gamma) ;
        }
    }
    return this->result_img;
}
cv::Mat Enhancement::logTransform(double  c){
    this->result_img=this->raw_image.clone();
    this->result_img.convertTo(this->result_img,CV_64F);
    for (int i=0;i<this->raw_image.rows;i++)
    {
        for (int j=0;j<this->raw_image.cols;j++){
            this->result_img.at<double >(i,j)= c*log2(1+(this->raw_image.at<uchar>(i,j))/255.0) ;
            // std::cout << this->result_img.at<double >(i,j) << std::endl;
        }
    }
    this->result_img*=255;
    this->result_img.convertTo(this->result_img,CV_8U);
    return this->result_img;
}

cv::Mat Enhancement::logTransform_f(double  c){
    this->result_img=this->raw_image.clone();
    this->result_img.convertTo(this->result_img,CV_64F);
    for (int i=0;i<this->raw_image.rows;i++)
    {
        for (int j=0;j<this->raw_image.cols;j++){
            this->result_img.at<double >(i,j)= c*log2(1+(this->raw_image.at<double>(i,j)) ) ;
        }
    }
    return this->result_img;
}