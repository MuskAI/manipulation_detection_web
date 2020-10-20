"""
Created by HAORAN
TIME 10/20
DESCRIPTION:
1. 一个GetScore class
2. 通过预测图返回文字性的说明
"""
import os
import numpy as np
from PIL import Image
import traceback
import sys
class GetScore():
    def __init__(self, img_path):
        """
        the input is a image path, first , we need path check
        channel number requirement: 1
        :param img:
        """
        if os.path.exists(img_path):
            print('路径合法')
            self.img = Image.open(img_path)
        else:
            print('路径不合法')
            traceback.print_exc()

    def basic_description(self):
        """
        计算公式:
        :return:
        """
        img_size = 320
        threshold1 = 0.6 * 255
        threshold2 = 0.95 * 255
        img = np.array(self.img)
        if img.size > img_size*img_size:
            print('the input error')
            return 'channel error'
        else:
            total_num = img.size
            img_threshold1 = np.where(img > threshold1, 255, 0)
            img_threshold2 = np.where(img > threshold2, 255, 0)

            if abs(sum(sum(np.where(img < 0.2 * 255, 1, 0))) - total_num) < 5:
                confidence = 0

            else:
                number_fenmu = int(sum(sum(img_threshold1))/255)
                number_fenzi = int(sum(sum(img_threshold2))/255)
                print(number_fenmu,number_fenmu)
                confidence = number_fenzi/number_fenmu

            return confidence

if __name__ == '__main__':
    """
    Usage：
    1. score = GetScore(pred_path).basic_description()
    """
    pred_path = 'C:\\Users\\musk\\Desktop\\compress1\\CompressedImageCM\\c_图片12.png'
    score = GetScore(pred_path).basic_description()