"""
created by haoran
time: 9/25
description:
这是一个调用本地GPU python环境，keras，tensorflow，进行模型代码的测试，并返回测试结果的API

Input: an image path
Ouput: an detection image path
"""

from keras import backend as K
import numpy as np
import glob
from PIL import Image
import cv2
from .U_Net_Down_stage_output import superPixNet
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def get_train_batch(batch_size,times,image_names,dir_path):
    batch_images_path = []
    selected_imgs_indexes=[batch_size*times+i for i in range(batch_size)]
    for index in selected_imgs_indexes:
        current_img =os.path.join(dir_path,image_names[index])
        batch_images_path.append(current_img)
    images ,path_list = get_img_and_labels(batch_images_path)
    return images ,path_list
def get_img_and_labels(batch_images_path):
    x_batch = []
    for image in batch_images_path:
        # image = image.split('/')[-1]
        name = image.split('/')[-1]
        im = Image.open(image)
        im = im.crop((0, 0, 320, 320))
        im = np.array(im, dtype=np.float32)
        im = im[..., ::-1]  # RGB 2 BGR
        R = im[..., 0].mean()
        G = im[..., 1].mean()
        B = im[..., 2].mean()
        im[..., 0] -= R
        im[..., 1] -= G
        im[..., 2] -= B
        x_batch.append(im)
    x_batch = np.array(x_batch, np.float32)
    return x_batch, batch_images_path
def testEnterFunction(input_dir, output_dir, model_path='/home/liu/chenhaoran/manipulation_detection_web/tensorflow_model/checkpoint/checkpoint.61-0.0452-0.9976-0.9341-0.8720-0.9004.hdf5'):
    print("####")
    print(input_dir)
    print(output_dir)
    print(model_path)
    K.set_image_data_format('channels_last')
    K.clear_session()
    K.image_data_format()
    model = superPixNet(input_shape=(320, 320, 3))
    model.load_weights(model_path, by_name=True)
    print('load weights success')
    # images, path_list = get_train_batch(batch_size=1, times=0, image_names= '', dir_path=dir_path)
    input_path_list = []
    input_path_list.append(input_dir)
    images , _ = get_img_and_labels(input_path_list)
    prediction = model.predict(images, batch_size=1)
    print(len(prediction))
    for i in range(len(prediction[0])):
        mask = np.zeros_like(images[i][:, :, :1])
        mask += (prediction[-1][i]) * 255
        final_save_path = output_dir.replace('.jpg','.png')
        print('save success',final_save_path)
        cv2.imwrite(final_save_path, mask)
        # 假彩色图
        im_gray = cv2.imread(final_save_path, cv2.IMREAD_GRAYSCALE)
        im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        cv2.imwrite(final_save_path.replace('output','im_output'),im_color)


if __name__ == "__main__":
    # main()
    testEnterFunction('../upload_dir/input/7.png','../upload_dir/output/7.png',model_path = './checkpoint/checkpoint.61-0.0452-0.9976-0.9341-0.8720-0.9004.hdf5')
    pass
