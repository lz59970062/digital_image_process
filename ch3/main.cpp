#include <iostream>
#include <opencv2/opencv.hpp>
#include "myfunc.h"

#define ITEM 2
int main(int arvc, char *argv[])
{

    Mat img;
#if ITEM == 1
    for (float i = 0; i < 0.8; i += 0.2)
    {
        img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\ch3\\PeppersRGB.bmp");
        saltandpepper(img, i);
        imshow("img", img);
        waitKey(0);
    }
#elif  ITEM==2
    img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\ch3\\PeppersRGB.bmp");
    Noise noise(img, "Gaussian");
    Mat img2=noise.gaussian(0, 50);
    hconcat(img, img2, img);
    imshow("img", img);
    
    waitKey(0);

#elif ITEM==3 
for(int i=0;i<100;i++){
    cout<<generateGaussianNoise(1,0.1)<<endl;

}


#endif
    return 0;
}
