#ifndef __MY_FUNC__
#define __MY_FUNC__
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;
void slatNosise(cv::Mat &img, double f);
cv::Mat img_transf(Mat &img, Mat &T, int methed);
Mat Transform(double theta, double sw, double sh, double mx, double my);
Mat img_inverse(Mat &img);
void saltandpepper(cv::Mat &image, float f);
Mat img_binary(Mat &img, uint8_t thr);
double generateGaussianNoise(double mu, double sigma);

class F_tr
{
public:
    F_tr(Mat & img)
    {
        this->img = img.clone();
    }
    
    Mat T;
    Mat mdft();
    Mat midft();
    Mat mdft_c(int met=0);
    Mat midft_c();
    Mat dft_r;
    Mat dft_i;
    Mat mdft_phase();
   
private:
    Mat img;
     
    
};

class Noise
{
    public:
    Noise(Mat & img, string name)
    {
        this->img = img.clone();
        this->name = name;
    }
    Mat saltandpepper(float f);
    Mat gaussian(float mean, float sigma);
    Mat rayleigh(float sigma);
    Mat exponential(float lambda);

    private:
    string name;
    Mat img;
    Mat noise_rate;
};
#endif