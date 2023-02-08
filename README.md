<!--
 * @Author: zhangqj zhangqj@shanghaitech.edu.cn
 * @Date: 2022-12-28 20:40:49
 * @LastEditors: zhangqj zhangqj@shanghaitech.edu.cn
 * @LastEditTime: 2022-12-28 21:26:29
 * @FilePath: /Skeleton-Based-Action-Quality-Assessment/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

This is the course project for CS272 Computer Vision II at [ShanghaiTech Univeristy][shanghaitech]
![Our Method](./dia1.png)
## Before Training
+ Download [FineDiving][finediving] and put it in this file.

+ Set up enviroment
```bash
cd DeblurGAN_pytorch
pip install -r requirements.txt
cd ../st-gcn
pip install -r requirements.txt
cd pytorchlight
python3 setup.py install
cd ../..
pip install -r requirements.txt
```



## Extracting Skeleton

+ Inaccurate (but fast) extracting with DeblurGAN 
        
        1.Before you run extract_skeleton.py, please downdown the Deblur weight: https://drive.google.com/open?id=1w-u0r3hd3cfzSjFuvvuYAs9wA-E-B-11

```bash
cd DeblurGAN_pytorch
python extract_skeleton.py
# or
./run.sh
```
+ Accurate extracting with YoloV7
      
  download the weight we trained [link][best]
```bash
cd yolov7
python detect.py --weights best.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```
        + or training with your own model
```bash
cd yolov7-pose-train
python train.py 
```
since the extracting takes long time(like 10 hours), you can also download the skeleton we provide(extracting from FineDiving):[link][sk]
## Training your model 
+ Action Quality Accessment-> you can set your own experiment refer to config.

```bash
cd st-gcn
# prepare data
# split data by data/spilt_dataset.py
# and gendata .npy by tools/fds_gendata.py
./prepare_data
# start_training or see other shell examples
./run.sh
```

## Model Weight
st-gcn + single:[link][1]

st-gcn+i3d + single:[link][2]

st-gcn + multi:[link][3]

st-gcn+i3d + multi:[link][4]

download the one you need and modify corresponding setting in config.

## Acknowledgements
This project is largely based on implementation of [i3d][i3d], [st-gcn][st-gcn], [musdl][musdl].


[shanghaitech]: https://www.shanghaitech.edu.cn/
[finediving]:https://github.com/xujinglin/FineDiving
[1]:https://drive.google.com/file/d/1cSxjvVXfIMNbqu8-n_uxl1rdD5wm1jkM/view?usp=share_link
[2]:https://drive.google.com/file/d/1iKaBotrwtNj081ino1tV1AFwopc7tuiv/view?usp=share_link
[3]:https://drive.google.com/file/d/1LEpGyZuZ-896BcjJOSoK-NeSnnFGwP8-/view?usp=share_link
[4]:https://drive.google.com/file/d/1btSVWOfFxquWmLdJKvnOA0PWTbpEDbRm/view?usp=share_link
[sk]:https://drive.google.com/drive/folders/1zoRgpX-fKkxp3hY8lYq73a7zfHU-6zQX?usp=share_link
[best]:https://drive.google.com/file/d/1-K-34Uxb6gc5MPc45XUmzRlIB-BQewDG/view?usp=share_link
[i3d]:https://github.com/PPPrior/i3d-pytorch
[st-gcn]:https://github.com/yysijie/st-gcn
[musdl]:https://github.com/nzl-thu/MUSDL
