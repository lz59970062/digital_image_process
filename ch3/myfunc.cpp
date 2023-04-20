
#include "myfunc.h"
#include <random>

inline void slatNosise(cv::Mat &img, float f)
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
inline Mat img_binary(Mat &img, uint8_t thr)
{
    if (img.empty())
    {
        std::cout << "img is empty" << std::endl;
        return img;
    }
    if (img.type() != CV_8UC1)
    {
        std::cout << "img type is not CV_8UC1,unsupport opt" << std::endl;
        return img;
    }
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<uchar>(i, j) > thr)
            {
                img.at<uchar>(i, j) = 255;
            }
            else
            {
                img.at<uchar>(i, j) = 0;
            }
        }
    }
    return img;
}

inline Mat img_inverse(Mat &img)
{
    if (img.empty())
    {
        std::cout << "img is empty" << std::endl;
        return img;
    }
    for (int i = 0; i < img.rows; i++)
    {
        void *ptr;
        if (img.channels() == 3)
        {
            ptr = img.ptr<Vec3b>(i);
        }
        else if (img.channels() == 1)
        {
            ptr = img.ptr<uchar>(i);
        }
        for (int j = 0; j < img.cols; j++)
        {

            if (img.channels() == 3)
            {
                ((Vec3b *)ptr)[j] = Vec3b(255, 255, 255) - ((Vec3b *)ptr)[j];
            }
            if (img.channels() == 1)
            {
                ((uchar *)ptr)[j] = 255 - ((uchar *)ptr)[j];
            }
        }
    }
    return img;
}

inline void addSaltNoise(Mat &img, int n)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> row_dist(0, img.rows - 1);
    std::uniform_int_distribution<int> col_dist(0, img.cols - 1);

    for (int k = 0; k < n; k++)
    {
        // Generate random row and column indices
        int i = row_dist(rng);
        int j = col_dist(rng);

        // Add salt noise to the image
        if (img.channels() == 1)
        {
            img.at<uchar>(i, j) = 255;
        }
        else if (img.channels() == 3)
        {
            img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
        }
    }
}

Vec3b inster1(const Mat &img, double x, double y) // 双线性插值
{
    double x1 = floor(x);
    double x2 = ceil(x);
    double y1 = floor(y);
    double y2 = ceil(y);

    // Clamp x2 and y2 to be within the valid range
    x2 = std::min(x2, img.cols - 1.0);
    y2 = std::min(y2, img.rows - 1.0);
    x1 = std::max(x1, 0.0);
    y1 = std::max(y1, 0.0);
    double dx = x - x1;
    double dy = y - y1;

    Vec3b p1 = img.at<Vec3b>(y1, x1);
    Vec3b p2 = img.at<Vec3b>(y1, x2);
    Vec3b p3 = img.at<Vec3b>(y2, x1);
    Vec3b p4 = img.at<Vec3b>(y2, x2);

    Vec3b r1 = p1 + (p2 - p1) * dx;
    Vec3b r2 = p3 + (p4 - p3) * dx;
    Vec3b r = r1 + (r2 - r1) * dy;

    return r;
}
Mat Transform(double theta, double sw, double sh, double mx, double my)
{
    Mat T = Mat::eye(3, 3, CV_32F);
    T.at<float>(0, 0) = cos(theta);
    T.at<float>(0, 1) = -sin(theta);
    T.at<float>(1, 0) = sin(theta);
    T.at<float>(1, 1) = cos(theta);
    T.at<float>(0, 2) = mx;
    T.at<float>(1, 2) = my;
    T.at<float>(0, 0) *= sw;
    T.at<float>(0, 1) *= sw;
    T.at<float>(1, 0) *= sh;
    T.at<float>(1, 1) *= sh;
    return T;
}

///////img_transformer///////
// arg img:img to trans
// T :mat transform mat
/*
|aR t|
|0  1|
a:scal mat
t:transport vec
R:rotation mat

*/
Mat img_transf(Mat &img, Mat &T, int methed)
{
    Mat R = T(Rect2d(0, 0, 2, 2));
    Mat t(2, 1, CV_32F);
    t.at<float>(0) = T.at<float>(0, 2);
    t.at<float>(1) = T.at<float>(1, 2);

    // culate new image size
    float data[3][2] = {{0.0, (float)img.cols}, {(float)img.cols, (float)img.rows}, {(float)img.rows, 0}};
    Mat p0 = R * Mat(2, 1, CV_32F, data[0]);
    Mat p1 = R * Mat(2, 1, CV_32F, data[1]);
    Mat p3 = R * Mat(2, 1, CV_32F, data[2]);
    Mat temp1 = p3 - p0;
    int new_w = abs(temp1.at<float>(0)) > abs(p1.at<float>(0)) ? abs(temp1.at<float>(0)) : abs(p1.at<float>(0));
    int new_h = abs(temp1.at<float>(1)) > abs(p1.at<float>(1)) ? abs(temp1.at<float>(1)) : abs(p1.at<float>(1));
    std::cout << new_w << " " << new_h << std::endl;
    Mat img2(new_w, new_h, CV_8UC3);
    Mat Tinv;

    // hconcat(R.t(), -t, Tinv);
    // float data2[] = {0.0, 0.0, 1.0};
    // vconcat(Tinv, Mat(1, 3, CV_32F, data2), Tinv);
    if (methed < 2)
    {
        Tinv = T.inv();

        for (auto i = 0; i < img2.rows; i++)
        {
            auto ptr = img2.ptr<Vec3b>(i);
            for (auto j = 0; j < img2.cols; j++)
            {
                Mat p = (Mat_<float>(3, 1) << j, i, 1);
                Mat p2 = Tinv * p;
                switch (methed)
                {
                case (0):

                    if (p2.at<float>(0) > 0 && p2.at<float>(1) > 0 && p2.at<float>(0) < img.cols && p2.at<float>(1) < img.rows)
                    {
                        ptr[j] = inster1(img, p2.at<float>(0), p2.at<float>(1)); // 双线性插值
                    }
                    else
                    {
                        ptr[j] = Vec3b(0, 0, 0);
                    }
                    break;
                case (1):

                    if (p2.at<float>(0) > 0 && p2.at<float>(1) > 0 && p2.at<float>(0) < img.cols && p2.at<float>(1) < img.rows)
                    {
                        ptr[j] = img.at<Vec3b>(round(p2.at<float>(1)), round(p2.at<float>(0))); // 最近邻插值
                    }
                    else
                    {
                        ptr[j] = Vec3b(0, 0, 0);
                    }
                    break;
                }
            }
        }
    }
    else if (methed == 2) // 前向映射
    {
        for (auto i = 0; i < img.rows; i++)
        {
            auto ptr = img.ptr<Vec3b>(i);
            for (auto j = 0; j < img.cols; j++)
            {
                Mat p = (Mat_<float>(3, 1) << j, i, 1);
                Mat p3 = T * p;
                if (p3.at<float>(0) > 0 && p3.at<float>(1) > 0 && p3.at<float>(0) < img2.cols && p3.at<float>(1) < img2.rows)
                {
                    img2.at<Vec3b>(p3.at<float>(1), p3.at<float>(0)) = ptr[j];
                }
            }
        }
    }

    return img2;
}

void saltandpepper(cv::Mat &image, float f)
{
    int n1 = image.rows * image.cols * f / 2;
    int n2 = image.rows * image.cols * f / 2;
    for (int k = 0; k < n1; k++)
    {
        int i = rand() % image.cols;
        int j = rand() % image.rows;
        if (image.channels() == 1)
        {
            image.at<uchar>(j, i) = 255;
        }
        else if (image.channels() == 3)
        {

            image.at<cv::Vec3b>(j, i)[0] = 255;
            image.at<cv::Vec3b>(j, i)[1] = 255;
            image.at<cv::Vec3b>(j, i)[2] = 255;
        }
    }
    for (int k = 0; k < n2; k++)
    {
        int i = rand() % image.cols;
        int j = rand() % image.rows;
        if (image.channels() == 1)
        {
            image.at<uchar>(j, i) = 0;
        }
        else if (image.channels() == 3)
        {
            image.at<cv::Vec3b>(j, i)[0] = 0;
            image.at<cv::Vec3b>(j, i)[1] = 0;
            image.at<cv::Vec3b>(j, i)[2] = 0;
        }
    }
}

Mat F_tr::mdft()
{
    Mat I = this->img.clone();

    Mat padded; // 以0填充输入图像矩阵
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols);
    // 填充输入图像I，输入矩阵为padded，上方和左方不做填充处理
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI); // 将planes融合合并成一个多通道数组complexI
    dft(complexI, complexI);    // 进行傅里叶变换
    // 计算幅值，转换到对数尺度(logarithmic scale)
    //=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes); // planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
                             // 即planes[0]为实部,planes[1]为虚部
    this->dft_i = planes[1].clone();
    this->dft_r = planes[0].clone();
    magnitude(planes[0], planes[1], planes[0]); // planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1);
    log(magI, magI); // 转换到对数尺度(logarithmic scale)
    return magI;
}
Mat F_tr::mdft_phase()
{
    Mat p(this->dft_r.size(), CV_32F);
    phase(this->dft_r, this->dft_i, p);

    normalize(p, p, 0, 255, NORM_MINMAX, CV_8UC1);
    return p;
}
Mat F_tr::mdft_c(int met)
{
    // 如果有奇数行或列，则对频谱进行裁剪
    if (met == 0)
    {
        Mat magI = mdft();
        magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
        // 重新排列傅里叶图像中的象限，使得原点位于图像中心
        int cx = magI.cols / 2;
        int cy = magI.rows / 2;
        Mat q0(magI, Rect(0, 0, cx, cy));   // 左上角图像划定ROI区域
        Mat q1(magI, Rect(cx, 0, cx, cy));  // 右上角图像
        Mat q2(magI, Rect(0, cy, cx, cy));  // 左下角图像
        Mat q3(magI, Rect(cx, cy, cx, cy)); // 右下角图像
        // 变换左上角和右下角象限
        Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        // 变换右上角和左下角象限
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
        // 归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
        normalize(magI, magI, 0, 1, NORM_MINMAX);
        // imshow("输入图像", I);
        // imshow("频谱图", magI);
        // waitKey(0);
        return magI;
    }
    else if (met == 1)
    {
        Mat I = this->img.clone();
        int w = I.cols;
        int h = I.rows;
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                I.at<uchar>(i, j) = I.at<uchar>(i, j) * pow(-1, i + j);
            }
        }
        F_tr f(I);
        Mat magI = f.mdft();
        normalize(magI, magI, 0, 1, NORM_MINMAX);
        return magI;
    }
}

// class Noise
// {
//     public:
//     Noise(Mat & img, string name)
//     {
//         this->img = img.clone();
//         this->name = name;
//     }
//     Mat saltandpepper(float f);
//     Mat gaussian(float mean, float sigma);
//     Mat rayleigh(float sigma);
//     Mat exponential(float lambda);

//     private:
//     string name;
//     Mat img;
//     Mat noise_rate;

Mat Noise::saltandpepper(float f)
{
    Mat image = this->img.clone();
    int n1 = image.rows * image.cols * f / 2;
    int n2 = image.rows * image.cols * f / 2;
    for (int k = 0; k < n1; k++)
    {
        int i = rand() % image.cols;
        int j = rand() % image.rows;
        if (image.channels() == 1)
        {
            image.at<uchar>(j, i) = 255;
        }
        else if (image.channels() == 3)
        {

            image.at<cv::Vec3b>(j, i)[0] = 255;
            image.at<cv::Vec3b>(j, i)[1] = 255;
            image.at<cv::Vec3b>(j, i)[2] = 255;
        }
    }
    for (int k = 0; k < n2; k++)
    {
        int i = rand() % image.cols;
        int j = rand() % image.rows;
        if (image.channels() == 1)
        {
            image.at<uchar>(j, i) = 0;
        }
        else if (image.channels() == 3)
        {
            image.at<cv::Vec3b>(j, i)[0] = 0;
            image.at<cv::Vec3b>(j, i)[1] = 0;
            image.at<cv::Vec3b>(j, i)[2] = 0;
        }
    }
    return image;
}

// 生成高斯分布的噪声：
vector<Vec3b> generateGaussianNoise(double mu, double sigma)
{
    // 定义一个特别小的值
    const double epsilon = numeric_limits<
        double>::min(); // 返回目标数据类型能表示的最逼近 1 的正数和 1 的差的绝对值
    double z0, z1;

    double u1, u2;
    double data[6];
    // 构造随机变量
    for (int i = 0; i < 3; i++)
    {
        do
        {
            u1 = rand() * (1.0 / RAND_MAX);
            u2 = rand() * (1.0 / RAND_MAX);
        } while (u1 <= epsilon);
        // flag 为真构造高斯随机变量 X
        z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
        z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
        data[2 * i] = z0 * sigma + mu;
        data[2 * i + 1] = z1 * sigma + mu;
        // 返回高斯随机变量
    }
    vector<Vec3b> noise;
    noise.push_back(Vec3d(data[0], data[1], data[2]));
    noise.push_back(Vec3d(data[3], data[4], data[5]));
    return noise;
}
Mat Noise::gaussian(float mean, float sigma)
{
    Mat image = this->img.clone();
    int channels = image.channels();
    int cols = image.cols;
    int rows = image.rows;
    if (image.isContinuous())
    {
        cols = cols * rows;
        rows = 1;
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j+=2)
        {
          vector<Vec3b> noise = generateGaussianNoise(mean, sigma);
          image.at<Vec3b>(i,j)= image.at<Vec3b>(i,j)+noise[0];
          image.at<Vec3b>(i,j+1)= image.at<Vec3b>(i,j+1)+noise[1];
        }
    }
    return image;
}
// 为图像添加高斯噪声
// Mat addGaussianNoise(Mat &srcImage)
// {
//     Mat resultImage = srcImage.clone();
//     // 深拷贝,克隆
//     int channels = resultImage.channels();
//     // 获取图像的通道
//     int nRows = resultImage.rows;            // 图像的行数
//     int nCols = resultImage.cols * channels; // 图像的总列数
//     // 判断图像的连续性
//     if (resultImage.isContinuous()) // 判断矩阵是否连续，若连续，我们相当于只需要遍历一个一维数组
//     {
//         nCols *= nRows;
//         nRows = 1;
//     }
//     for (int i = 0; i < nRows; i++)
//     {
//         for (int j = 0; j < nCols; j++)
//         { // 添加高斯噪声
//             int val = resultImage.ptr<uchar>(i)[j] +
//                       generateGaussianNoise(2, 0.8) * 32;
//             if (val < 0)
//                 val = 0;
//             if (val > 255)
//                 val = 255;
//             resultImage.ptr<uchar>(i)[j] =
//                 (uchar)val;
//         }
//     }
//     return resultImage;
// }
// int main()
// {
// Mat srcImage = imread("D:\\1.jpg");
// if (!srcImage.data)
// return -1;
// imshow("srcImage", srcImage);
// Mat resultImage = addGaussianNoise(src
// Image);
// imshow("resultImage", resultImage);
// waitKey(0);
// return 0;
// }