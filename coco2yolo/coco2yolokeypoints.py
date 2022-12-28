# COCO 格式的数据集转化为 YOLO 格式的数据集
# --json_path 输入的json文件路径
# --save_path 保存的文件夹名字，默认为当前目录下的labels。
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
from j2t import *

parser = argparse.ArgumentParser()
# 这里根据自己的json文件位置，换成自己的就行
parser.add_argument('--json_path', default=r'/home/p300/cs272/project/coco_kpts/annotations/val.json',
                    type=str,
                    help="input: coco format(json)")
# 这里设置.txt文件保存位置
parser.add_argument('--save_path', default=r'/home/p300/cs272/project/coco_kpts/annotations/val_n', type=str,
                    help="specify where to save the output dir of labels")
arg = parser.parse_args()

xiaoshu = 10 ** 6  # 保存几位小数, 6位
cun = r'/home/p300/cs272/project/coco_kpts/annotations/val_n'

if __name__ == '__main__':
    json_file = arg.json_path  # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    # print(id_map)
    # 这里需要根据自己的需要，更改写入图像相对路径的文件位置。
    list_file = open(os.path.join(ana_txt_save_path, 'val.txt'), 'w')

    for annotations in tqdm(data['annotations']):
        print(annotations)
        # filename = annotations["file_name"]
        img_width = 455#annotations["width"]
        img_height = 256#annotations["height"]
        img_id = annotations["image_id"]
        keypoints = annotations["keypoints"]

        # print(keypoints)
        arry_x = np.zeros([17, 1])
        num_1 = 0
        for x in keypoints[0:51:3]:
            arry_x[num_1, 0] = int((x / img_width) * xiaoshu) / xiaoshu
            num_1 += 1

        arry_y = np.zeros([17, 1])
        num_2 = 0
        for y in keypoints[1:51:3]:
            arry_y[num_2, 0] = int((y / img_height) * xiaoshu) / xiaoshu
            num_2 += 1

        arry_v = np.zeros([17, 1])
        num_3 = 0
        for v in keypoints[2:51:3]:
            arry_v[num_3, 0] = v
            num_3 += 1

        list_1 = []
        num_4 = 0
        for i in range(17):
            list_1.append(float(arry_x[num_4]))
            list_1.append(float(arry_y[num_4]))
            list_1.append(float(arry_v[num_4]))
            num_4 += 1


        fil = img_id

        cun_1 = os.path.join(cun, str(fil))
        cun_2 = cun_1 + ".txt"

        with open(cun_2, "r") as f:
            read_lines = f.readlines()
        x_1 = read_lines[0].strip().split(",")
        x_2 = x_1 + list_1
        list_xin = list(map(str, x_2))
        list_xin = " ".join(list_xin)
        with open(cun_2, "w") as f:
            f.write(list_xin)

    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        os.chdir(cun)
        # os.rename(ana_txt_name, str(img_id) + ".txt")
        os.rename(str(img_id) + ".txt", ana_txt_name)

