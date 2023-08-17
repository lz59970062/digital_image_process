void ruihua(cv::Mat &image, cv::Mat &result, int type)
{
    result.create(image.size(), image.type());
    // if (image.channels() != 1)
    //  return;
    int w = image.cols * image.channels();
    int h = image.rows;
    int y, x;
    switch (type)
    {
    case 0:
        for (y = 0; y < h - 1; y++)
        {
            unsigned char *lpsrc = image.ptr<uchar>(y);
            unsigned char *lpsrc1 = image.ptr<uchar>(y + 1);
            for (x = 0; x < w - 1; x++)
            {
                float temp = abs(lpsrc[x] - lpsrc[x + 1]) + abs(lpsrc1[x] -
                                                                lpsrc[x]);
                result.data[y * w + x] = cv::saturate_cast<uchar>(temp);
            }
        }
        break;
    case 1:
        for (y = 0; y < h - 1; y++)
        {
            unsigned char *lpsrc = image.ptr<uchar>(y);
            unsigned char *lpsrc1 = image.ptr<uchar>(y + 1);
            for (x = 0; x < w - 1; x++)
            {
                float temp = abs(lpsrc[x] - lpsrc1[x + 1]) +
                             abs(lpsrc1[x] - lpsrc[x + 1]);
                result.data[y * w + x] = cv::saturate_cast<uchar>(temp);
            }
        }
        break;
    case 2:
        for (y = 1; y < h - 1; y++)
        {
            unsigned char *lpsrc = image.ptr<uchar>(y);
            unsigned char *lpsrc1 = image.ptr<uchar>(y + 1);
            uchar *lpsrc2 = image.ptr<uchar>(y - 1);
            for (x = 1; x < w - 1; x++)
            {
                float sx = 2 * (lpsrc[x + 1] - lpsrc[x - 1]) + (lpsrc1[x + 1] - lpsrc1[x - 1]) +
                           (lpsrc2[x + 1] - lpsrc2[x - 1]);
                float sy = 2 * (lpsrc2[x] - lpsrc1[x]) + (lpsrc2[x + 1] - lpsrc1[x + 1]) +
                           (lpsrc2[x - 1] - lpsrc1[x - 1]);
                uchar temp = cv::saturate_cast<uchar>(abs(sx) + abs(sy));
                result.data[y * w + x] = temp;
            }
        }
        break;
    case 3:
        for (y = 0; y < h - 1; y++)
        {
            unsigned char *lpsrc = image.ptr<uchar>(y);
            unsigned char *lpsrcnext = image.ptr<uchar>(y + 1);
            unsigned char *lpsrcpre = image.ptr<uchar>(y - 1);
            for (x = 0; x < w - 1; x++)
            {
                int fxy = lpsrc[x];
                int fxy0 = lpsrcpre[x];
                int fxy1 = lpsrcnext[x];
                int fxy2 = lpsrc[x - 1];
                int fxy3 = lpsrc[x + 1];
                int lpls = fxy0 + fxy1 + fxy2 + fxy3 - 4 * fxy;
                result.data[y * w + x] = cv::saturate_cast<uchar>(fxylpls);
            }
        }
        break;
    default:
        break;
    }
}