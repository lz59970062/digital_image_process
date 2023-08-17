import cv2
import numpy as np
import time

# 测试10种不同尺寸的图片
for i in range(1, 11):
    width = 100 * i  # 将图片的宽度设置为100的倍数
    height = 100 * i  # 将图片的高度设置为100的倍数

    # 开始计时
    start_gen = time.time()

    # 创建一个随机填充的图像
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # 停止计时
    end_gen = time.time()

    # 开始计时
    start_blur = time.time()

    # 对图像进行高斯模糊
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 停止计时
    end_blur = time.time()

    # 计算并输出生成图像和高斯模糊所需的时间
    duration_gen = end_gen - start_gen
    duration_blur = end_blur - start_blur
    print(f"Image size: {width}x{height}, "
          f"Image generation time: {duration_gen:.6f} seconds, "
          f"Gaussian Blur time: {duration_blur:.6f} seconds")
