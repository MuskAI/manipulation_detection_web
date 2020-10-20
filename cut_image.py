from PIL import Image

# 处理图片函数
def cut_image(image):
    width, height = image.size
    item_width = 320
    box_list = []
    box_list2 = []
    box_list3 = []
    count = 0
    rest_width = 0
    rest_height = 0
    img1 = image
    img2 = image
    if width > 320:
        m = int(width/320)
        rest_width = width - m * 320
    else:
        m = 0
        rest_width = width
    if height > 320:
        n = int(height/320)
        rest_height = height - n * 320
    else:
        n = 0
        rest_height = height

    if(m == 0 and n == 0):
        count += 1
        image = image.resize((item_width, item_width), resample=0)
        image_list = [image.crop((0,0,item_width,item_width))]
    else:
        # 处理大块的图片
        for j in range(0, n):
            for i in range(0, m):
                count += 1
                box = (i * item_width, j * item_width, (i + 1) * item_width, (j + 1) * item_width)
                box_list.append(box)
        image_list = [image.crop(box) for box in box_list]

        # 处理垂直小块图片
        if (rest_width > 0):
            img1 = image.crop((m * item_width, 0, width, height))
            img1 = img1.resize((item_width, height), resample=0)
            for i in range(0, n):
                count += 1
                box = (0, i * item_width, item_width, (i + 1) * item_width)
                box_list2.append(box)
            if (rest_height > 0):
                count += 1
                box = (0, n * item_width, item_width, (n + 1) * item_width)
                box_list2.append(box)
        for box in box_list2:
            image_list.append(img1.crop(box))

        # 处理水平小块图片
        if (rest_height > 0):
            img2 = image.crop((0, n * item_width, m * width, height))
            img2 = img2.resize((width - rest_width, item_width), resample=0)
            for i in range(0, m):
                count += 1
                box = (i * item_width, 0, (i + 1) * item_width, item_width)
                box_list3.append(box)
        for box in box_list3:
            image_list.append(img2.crop(box))

    print(count)
    return image_list

# 输出文件函数
def save_images(image_list,save_file_path):
    index = 1
    for image in image_list:
        image.save(save_file_path + '\\' + str(index) + '.png')
        index += 1


if __name__ == '__main__':
    # 输入输出文件地址
    open_file_path = r'C:\Users\aaa\Desktop\pic\13.png'
    save_file_path = r'C:\Users\aaa\Desktop\pic'
    # 打开图像
    image = Image.open(open_file_path)
    # 分为图像
    image_list = cut_image(image)
    # 保存图像
    save_images(image_list,save_file_path)
