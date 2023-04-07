
#include "myfunc.h"
#include <random>

void slatNosise(cv::Mat &img, float f)
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
Mat img_binary(Mat &img, uint8_t thr)
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

Mat img_inverse(Mat &img)
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

void addSaltNoise(Mat &img, int n)
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

Vec3b inster(const Mat &img, double x, double y)
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
// methed 0 None 1:center

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
    if (methed == 1)
    {
        t.at<float>(0) += new_w / 2;
        t.at<float>(1) += new_h / 2;
    }
    // hconcat(R.t(), -t, Tinv);
    // float data2[] = {0.0, 0.0, 1.0};
    // vconcat(Tinv, Mat(1, 3, CV_32F, data2), Tinv);
    Tinv = T.inv();

    for (auto i = 0; i < img2.rows; i++)
    {
        auto ptr = img2.ptr<Vec3b>(i);
        for (auto j = 0; j < img2.cols; j++)
        {
            Mat p = (Mat_<float>(3, 1) << j, i, 1);
            Mat p2 = Tinv * p;

            if (p2.at<float>(0) > 0 && p2.at<float>(1) > 0 && p2.at<float>(0) < img.cols && p2.at<float>(1) < img.rows)
            {
                ptr[j] = inster(img, p2.at<float>(0), p2.at<float>(1));
            }
            else
            {
                ptr[j] = Vec3b(0, 0, 0);
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