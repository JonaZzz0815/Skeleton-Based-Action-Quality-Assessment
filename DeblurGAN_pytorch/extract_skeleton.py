import os
from PoseDetection import PoseDetector
import cv2
import numpy as np
from DeblurGANPredictor import Deblurer
import time
from tqdm import tqdm
import gc
import argparse
Deblur = Deblurer()

def extract_from_img(img_path,detector,deblur=True)->list:
    '''
    input: the image path 
    output: landmark
    '''
    if deblur : 
        image = Deblur.deblur(img_path)
    else:
        image = cv2.imread(img_path)

    _,landmarks,_=detector.findPose(image, draw=False, display=False)
    # process the landmark into list
    if landmarks == None:
        return None
    lmk_list = []
    for mk in landmarks.landmark:
        lmk_list.append([str(mk.x), str(mk.y),
                        str(mk.visibility) ])

    return lmk_list


'''
expected data form for skeleton
# 
70   # number of frame in this clip
2     #从第二行开始分别为每一帧的信息,对于其中每一帧，第一个数字为当前帧body数量（如1或2）
72057594037944738 0 1 1 1 1 0 0.01119184 -0.256052 2  
# “没有必要遵循这个格式，随便搞一串数字就行了”
# body_info =[ 'bodyID', 'clipedEdges', 'handLeftConfidence',
#                  'handLeftState', 'handRightConfidence', 'handRightState',
#                 'isResticted', 'leanX', 'leanY', 'trackingState' ]*/
# len = 10
25    # “关节点，根据模型的关节点个数给就行了” 个数25个关节点 25*12  每个关节包含12个字段信息
# joint_info_key = [
#                     'x', 'y', 'z', 
#                     'depthX', 'depthY', 'colorX', 'colorY',
#                     'orientationW', 'orientationX', 'orientationY',
#                     'orientationZ', 'trackingState'
#                 ]
# "这里只要x,y,z即2d坐标和置信度z就行了，其他info——key可以随便搞一个数字就上去占位置"
0.7201789 0.1647426 3.561895 336.5812 192.8021 1184.537 489.4118 -0.2218947 0.03421978 0.968875 -0.1042736 2
........ #省略25行
# body 2
25
-0.387201 -0.05403496 2.961099 214.7408 216.41 834.7666 559.0992 0.4557798 -0.05701343 0.8802808 -0.1188276 2
........ #省略25行

'''
def give_meaningless_nums(num:int)->list:
    return [str(-1) for i in range(num)]

def extract_from_clip(clip_path, out_path,skip_path)->None:
    # open the aimed_file
    out = open(out_path, mode="w")
    
    detector = PoseDetector()
    img_names = os.listdir(clip_path)
    # Check if all of file in this dir ending with .jpg???

    
    # write the total number of frame
    # loop each frame
    # out.write(f"{len(skeletons)} \n")
    # skip cnt
    cnt = 0
    skip_frames = []
    for img_name in sorted(img_names):

        # extract all the skeletons
        img_path = os.path.join(clip_path,img_name)
        lmk = extract_from_img(img_path,detector)
        if lmk == None:
            skip_frames.append(img_name)
            continue
        cnt += 1
        
        # write number of body in this frame
        out.write(f"{1} \n")

        # write the number of joints
        num_joint = len(lmk)
        out.write(f"{num_joint} \n")
        # write the body info
        out.write(" ".join(give_meaningless_nums(10)) + "\n")
        # write each joint
        for joint in lmk:
            to_write = joint + give_meaningless_nums(9)
            out.write(" ".join(to_write) + "\n")
    out.close()
    gc.collect()
    # rewrite the file
    f = open(out_path,mode="r")
    a = f.readlines()
    f.close()

    f = open(out_path,mode="w")
    f.write(str(cnt)+"\n")
    f.writelines(a)
    f.close()
    # write to a skip_frame
    f = open(skip_path,"w")
    f.writelines(skip_frames)
    f.close()

    print(f"Done! Write to {out_path}")
    print(f"But skip with {len(img_names) - cnt}")
    print(f"Write to {skip_path}")

def mkdir(file_name):
    try:
        os.mkdir(file_name+"/")
    except:
        print(f"{file_name} existed!")

if __name__ == "__main__":
    dataset_path = "../FineDiving_Dataset/Trimmed_Video_Frames/FINADiving_MTL_256s"
    output_orgin = "../FineDiving_Skeleton_3"
    mkdir(output_orgin)
    # all competitions' name 
    competition_ids = sorted(os.listdir(dataset_path))
    length = len(competition_ids)
    parser= argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0)
    args = parser.parse_args()


    # competition_ids = competition_ids[int(length/2)+ 2:]
    # loop to extract skeleton from video

    # for comp in tqdm(competition_ids):
    #     # get the dir name of one comp
    #     dir_name = os.path.join(dataset_path,comp)

    #     # make the output dir name of one comp
    #     output_comp_dir = os.path.join(output_orgin,comp)
    #     mkdir(output_comp_dir)
    #     # get clips names
    #     clips = os.listdir(dir_name)

    #     for clip in tqdm(clips):
    #         clip_dir_name = os.path.join(dir_name,clip)
    #         out_path = os.path.join(output_comp_dir,clip) + ".txt"
    #         skip_path = os.path.join(output_comp_dir,clip) + "_skip.txt"
    #         if os.path.exists(out_path):
    #             continue
    #         extract_from_clip(clip_dir_name,out_path,skip_path)
    #     print("#"*16 + f"Done extrating {comp}" + "#"*16 )
    
    # for left-over or empty txt
    # print("Checking the empty txt")
    # for comp in tqdm(competition_ids):
    #     # get the dir name of one comp
    #     dir_name = os.path.join(dataset_path,comp)

    #     # make the output dir name of one comp
    #     output_comp_dir = os.path.join(output_orgin,comp)
    #     mkdir(output_comp_dir)
    #     # get clips names
    #     clips = os.listdir(dir_name)

    #     for clip in tqdm(clips):
    #         clip_dir_name = os.path.join(dir_name,clip)
    #         out_path = os.path.join(output_comp_dir,clip) + ".txt"
    #         f = open(out_path,"r")
    #         if len(f.readlines()) == 0:
    #             f.close()
    #             extract_from_clip(clip_dir_name,out_path,skip_path)
    #         else:
    #             f.close()

    print(f"Re-Process the empty txt")
    empty_txt_path = "../empty_txt.txt"
    f = open(empty_txt_path,"r")
    for name in f.readlines():
        comp_id,clip_id = name.split("/")[-2:]
        clip_id = clip_id.split(".")[0]
        clip_dir_name = os.path.join(dataset_path,comp_id,clip_id)
        out_path = os.path.join(output_orgin,comp_id,clip_id) + ".txt"
        skip_path = os.path.join(output_orgin,comp_id,clip_id) + "_skip.txt"
        extract_from_clip(clip_dir_name,out_path,skip_path)
    print("Done!")

