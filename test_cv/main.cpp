
// #include <opencv2/opencv.hpp>
// #include <iostream>

// using namespace cv;
// using namespace std;

// // 对图像频谱进行移位操作，使用频谱分布在中心
// void fftshift(cv::Mat &magImage)
// {
//     magImage = magImage(Rect(0, 0, magImage.cols & -2, magImage.rows & -2));
//     int cx = magImage.cols / 2;
//     int cy = magImage.rows / 2;
//     Mat q0(magImage, Rect(0, 0, cx, cy));
//     Mat q1(magImage, Rect(cx, 0, cx, cy));
//     Mat q2(magImage, Rect(0, cy, cx, cy));
//     Mat q3(magImage, Rect(cx, cy, cx, cy));
//     // 交换象限(左上与右下进行交换)
//     Mat tmp;
//     q0.copyTo(tmp);
//     q3.copyTo(q0);
//     tmp.copyTo(q3);
//     // 交换象限（右上与左下进行交换）
//     q1.copyTo(tmp);
//     q2.copyTo(q1);
//     tmp.copyTo(q2);
// }
// // 傅里叶变换
// Mat mydft(cv::Mat srcImage)
// {
//     // 2 将输入图像扩展到最佳尺寸,边界用 0 补充
//     int m = getOptimalDFTSize(srcImage.rows);
//     int n = getOptimalDFTSize(srcImage.cols);
//     // 将添加的像素初始化为 0
//     Mat padded;
//     copyMakeBorder(srcImage, padded, 0, m - srcImage.rows,
//                    0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
//     // 3 为傅里叶变换的结果(实部和虚部)分配存储空间
//     // 将数组组合合并为一个多通道数组
//     Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
//     Mat complexI;
//     merge(planes, 2, complexI);
//     // 4 进行傅里叶变换
//     dft(complexI, complexI);
//     // idft(complexI, complexI);
//     // 将复数转换为幅值，即=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//     // 将多通道数组分离为几个单通道数组
//     split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//     magnitude(planes[0], planes[1], planes[0]);
//     Mat magImage = planes[0];
//     // 对数尺度缩放
//     magImage += Scalar::all(1);
//     log(magImage, magImage); // 求自然对数
//     // 剪切和重分布幅度图象限
//     // 若有奇数行或奇数列，进行频谱剪裁
//     magImage = magImage(Rect(0, 0, magImage.cols & -2, magImage.rows & -2));
//     // 重新排列傅立叶图像中的象限，使得原点位于图像中心
//     fftshift(magImage);
//     // 归一化，用 0 到 1 的浮点值将矩阵变换为可视的图像格式
//     normalize(magImage, magImage, 0, 1, NORM_MINMAX);
//     return magImage;
// }

// // 理想低通滤波器
// Mat LowPass(cv::Mat srcImage, int D0)
// {
//     // 2 将输入图像扩展到最佳尺寸,边界用 0 补充
//     int m = getOptimalDFTSize(srcImage.rows);
//     int n = getOptimalDFTSize(srcImage.cols);
//     // 将添加的像素初始化为 0
//     Mat padded;
//     copyMakeBorder(srcImage, padded, 0, m - srcImage.rows,
//                    0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
//     // 3 为傅里叶变换的结果(实部和虚部)分配存储空间
//     // 将数组组合合并为一个多通道数组
//     Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
//     Mat complexI;
//     merge(planes, 2, complexI);
//     // 4 进行傅里叶变换
//     dft(complexI, complexI);
//     // idft(complexI, complexI);
//     // 将复数转换为幅值，即=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//     // 将多通道数组分离为几个单通道数组
//     split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//     // magnitude(planes[0], planes[1], planes[0]);
//     Mat Re;
//     Re = planes[0];
//     Mat Im;
//     Im = planes[1];
//     long h = complexI.rows;
//     long w = complexI.cols;
//     fftshift(planes[0]);
//     fftshift(planes[1]);
//     for (int y = 0; y < h; y++)
//         for (int x = 0; x < w; x++)
//         {
//             if ((y - h / 2) * (y - h / 2) + (x - w / 2) * (x - w / 2) > D0 * D0)
//             {
//                 planes[0].at<float>(y, x) = 0;
//                 planes[1].at<float>(y, x) = 0;
//             }
//         }
//     Mat complex2;
//     merge(planes, 2, complex2);
//     idft(complex2, complex2);
//     split(complex2, planes);
//     magnitude(planes[0], planes[1], planes[0]);
//     Mat magImage = planes[0].clone();
//     // 归一化，用 0 到 1 的浮点值将矩阵变换为可视的图像格式
//     normalize(magImage, magImage, 0, 1, NORM_MINMAX);
//     return magImage;
// }
// // 布特沃斯滤波器卷积核生成
// Mat butterworth_lbrf_kernel(Mat &scr, float sigma, int n)
// {
//     Mat butterworth_low_pass(scr.size(), CV_32FC1); // ，CV_32FC1
//     double D0 = sigma;                              // 半径 D0 越⼩，模糊越⼤；半径 D0 越⼤，模糊越⼩
//     for (int i = 0; i < scr.rows; i++)
//     {
//         for (int j = 0; j < scr.cols; j++)
//         {
//             double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2)); // 分⼦,计算 pow 必须为 float 型
//             butterworth_low_pass.at<float>(i, j) = float(1.0 / (1 + pow(d / D0, 2 * n)));
//         }
//     }
//     std::string name = "Butterworth LOW PASS d0=" + std::to_string(sigma) + "n=" +
//                        std::to_string(n);
//     // putText(butterworth_low_pass, name, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2, 8);
//     imshow("Butterworth LOW PASS", butterworth_low_pass);
//     return butterworth_low_pass;
// }
// // 使用卷积核进行频域滤波
// Mat freqfilt(Mat &scr, Mat &blur)
// {
//     //***********************DFT*******************
//     Mat planes[] = {scr, Mat::zeros(scr.size(), CV_32FC1)}; // 创建通道，存储 dft 后的实部与虚部（CV_32F，必须为单通道数）
//     Mat complexIm;
//     merge(planes, 2, complexIm); // 合并通道 （把两个矩阵合并为⼀个 2 通道的 Mat类容器）
//     dft(complexIm, complexIm);   // 进⾏傅⽴叶变换，结果保存在⾃⾝
//     split(complexIm, planes);    // 分离通道（数组分离）
//     fftshift(planes[0]);
//     fftshift(planes[1]);
//     //*****************滤波器函数与 DFT 结果的乘积****************
//     Mat blur_r, blur_i, BLUR;
//     multiply(planes[0], blur, blur_r); // 滤波（实部与滤波器模板对应元素相乘）
//     multiply(planes[1], blur, blur_i); // 滤波（虚部与滤波器模板对应元素相乘）
//     Mat plane1[] = {blur_r, blur_i};
//     merge(plane1, 2, BLUR);                             // 实部与虚部合并
//     idft(BLUR, BLUR);                                   // idft 结果也为复数
//     split(BLUR, planes);                                // 分离通道，主要获取通道
//     magnitude(planes[0], planes[1], planes[0]);         // 求幅值(模)
//     normalize(planes[0], planes[0], 1, 0, NORM_MINMAX); // 归⼀化便于显⽰
//     return planes[0];
// }
// Mat Butterworth_Low_Paass_Filter(Mat &src, float d0, int n)
// {
//     // H = 1 / (1+(D/D0)^2n) n 表⽰巴特沃斯滤波器的次数
//     // 调整图像加速傅⾥叶变换
//     int M = getOptimalDFTSize(src.rows);
//     int N = getOptimalDFTSize(src.cols);
//     Mat padded;
//     copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols,
//                    BORDER_CONSTANT, Scalar::all(0));
//     padded.convertTo(padded, CV_32FC1);                              // 将图像转换为 float 型
//     Mat butterworth_kernel = butterworth_lbrf_kernel(padded, d0, n); // 理想低通滤波器
//     Mat result = freqfilt(padded, butterworth_kernel);
//     return result;
// }
// #define ITEM 1
// int main()
// {
//     Mat img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\test_cv\\PeppersRGB.bmp", 0);
//     if (img.data == NULL)
//         return -1;
//     imshow("testimage", img);
//     Mat fre = mydft(img); // 生成频谱
//     imshow(" frequency spectrum", fre);
//     cv::namedWindow("Trackbar Window", 1);

//     // 创建一个变量来保存Trackbar的值
//     int value = 50;
//     int maxValue = 300;
//     int step=2;

//     // 创建Trackbar
//     cv::createTrackbar("Trackbar", "Trackbar Window", &value, maxValue);
//     cv::createTrackbar("step", "Trackbar Window", &step, 100);
//     while (1)
//     {
//         Mat result = Butterworth_Low_Paass_Filter(img, value, 10);
//     //   Mat   result = LowPass(img, value ); // 理想低通
//         cv::imshow("low pass filter ", result);
//         cv::waitKey(4);
//     }
//     return 0;
    
//     // Mat img = imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\test_cv\\fenjin.jpg", -1);
//     // imshow("图像低通滤波",result);
//     // result = LowPass(img, 30); // 理想低通
//     // imshow("图像低通滤波", result);
//     // waitKey(0);
// }

// // #include <opencv2/opencv.hpp>
// // #include <iostream>

// // using namespace cv;
// // using namespace std;


// // int main(){
// //     Mat img=imread("C:\\Users\\lz599\\Desktop\\digital_image_process\\test_cv\\PeppersRGB.bmp",0);
// //     imshow("原图",img);
// //     Mat img1;
// //     img.convertTo(img1,CV_32FC1);
// //     imshow("原图2",img1/255.0);
// //     waitKey(0);
// //     return 0;
// // }


#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void test_camera_fps(int camera_index=1)
{
    VideoCapture cap(camera_index);
    cap.set(CAP_PROP_FPS,120);
    cap.set(CAP_PROP_FRAME_WIDTH,640);
    cap.set(CAP_PROP_FRAME_HEIGHT,480);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (!cap.isOpened())
    {
        cout << "Error: Unable to open camera." << endl;
        return;
    }

    int frame_count = 0;
    double start_time = (double)getTickCount();
    double avg_fps = 0;
    int idx=0;

    while (true)
    {
        Mat frame;
        bool ret = cap.read(frame);
        if (!ret)
        {
            cout << "Error: Unable to read frame." << endl;
            break;
        }

        frame_count += 1;

        if (frame_count %16 == 0)
        {
            double elapsed_time = ((double)getTickCount() - start_time)/getTickFrequency();
            avg_fps = (frame_count / elapsed_time);
            start_time = (double)getTickCount();
            frame_count = 0;
        }

        putText(frame, format("Average FPS: %.2f", avg_fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        imshow("Camera Test", frame);


        if (waitKey(1) >27)
        {
            imwrite(format("img%d.jpg", idx), frame);
            idx+=1;
        }

    }

    cap.release();
    destroyAllWindows();

}

int main()
{
    test_camera_fps();
}