#pragma once // 防止头文件重复包含
#include <opencv2/opencv.hpp>
#include <iostream>
#define pi 3.1415926535897932384626433832795
using namespace std;
class Cimage_proc // 图像处理类
{
public:
    inline static int color2rgb(cv::Mat &colorimage, cv::Mat &r, cv::Mat &g,
                                cv::Mat &b)
    {
        if (colorimage.channels() < 3)
        {
            return -1;
        }
        int w = colorimage.cols;
        int h = colorimage.rows;
        vector<cv::Mat> channels  ;
        cv::split(colorimage, channels);
        r = channels[2];
        g = channels[1];
        b = channels[0];
        return 0;
 
    }
    // RGB到HIS：
    inline static bool rgb2hsi(double *H, double *S, double *I, double *R, double *G, double *B, long w, long h,bool scaled=true)
    {
        long i, j, k;
        double theta;
        double eps = 0.000000000001;
        for (i = 0; i < h; i++)
            for (j = 0; j < w; j++)
            {
                k = i * (w) + j;
                I[k] = (1.0 * (R[k] + G[k] + B[k])) / 3.0;//I值即为计算三通道平均值
                if(!scaled)
                I[k] = I[k] / 255;//归一化
                S[k] = 1.0 - 3.0 * std::min(R[k], std::min(G[k], B[k])) / (R[k] + G[k] + B[k] + eps);//S值计算是根据RGB三通道中最小值计算的
                double num = 0.5 * ((R[k] - G[k]) + (R[k] - B[k]));//计算H值的分子
                double den = sqrt((R[k] - G[k]) * (R[k] - G[k]) + (R[k] - B[k]) * (G[k] - B[k]));//计算H值的分母
                if (den == 0)//避免出现除零错误
                    den = eps;
                theta = acos(num / (den));//acos返回的是弧度值
                if (B[k] > G[k])//转换到0-2pi
                    H[k] = 2 * pi - theta;
                else
                    H[k] = theta;
                if (S[k] == 0)//做一些限制 
                    H[k] = 0;
                H[k] /= 2 * pi;//将结果转换到0-1
            }
        return true;
    }

    inline static bool hsi2rgb(double *R, double *G, double *B, double *H, double *S, double *I, long w, long h)
    {
        long i, j, k;
        double th, ts, ti, tr, tg, tb;
        for (i = 0; i < h; i++)
            for (j = 0; j < w; j++)
            {
                k = i * (w) + j;
                th = H[k] * 2 * pi;
                ts = S[k];
                ti = I[k];
                if ((th < 2 * pi / 3) && (th >= 0))
                {
                    tb = ti * (1 - ts);
                    tr = ti * (1 + ts * cos(th) / cos(pi / 3 - th));
                    tg = 3 * ti - (tr + tb);
                }
                else if (th < 4 * pi / 3)
                {
                    th = th - 2 * pi / 3;
                    tr = ti * (1 - ts);
                    tg = ti * (1 + ts * cos(th) / cos(pi / 3 - th));
                    tb = 3 * ti - (tr + tg);
                }
                else
                {
                    th = th - 4 * pi / 3;
                    tg = ti * (1 - ts);
                    tb = ti * (1 + ts * cos(th) / cos(pi / 3 - th));
                    tr = 3 * ti - (tg + tb);
                }
                if (tr < 0)
                    tr = 0;
                if (tr > 1)
                    tr = 1;
                if (tg < 0)
                    tg = 0;
                if (tg > 1)
                    tg = 1;
                if (tb < 0)
                    tb = 0;
                if (tb > 1)
                    tb = 1;
                R[k] = tr * 255;
                G[k] = tg * 255;
                B[k] = tb * 255;
            }
        return true;
    }
};