import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.fds_read_skeleton import read_xyz


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 1
num_joint = 33
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")

path = "../FineDiving_Dataset/Annotations/FineDiving_fine-grained_annotation.pkl"  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
anno = open(path, 'rb')
anno_data = pickle.load(anno)


path = '../FineDiving_Dataset/Annotations/FineDiving_coarse_annotation.pkl' # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f = open(path, 'rb')
simple_data = pickle.load(f)

def get_score(comp_id,clip_id):
    return anno_data[(comp_id,clip_id)]

FineDiving_path = "../FineDiving_Dataset/Trimmed_Video_Frames/FINADiving_MTL_256s"
def process_rgb(comp_id:str,clip_id:str):


    dir_path = os.path.join(FineDiving_path,comp_id,clip_id)
    imgs = sorted(os.listdir(dir_path))[-20:]
    # print(type(imgs))
    entry_clip  = [ os.path.join(dir_path,img) 
                    for img in list(imgs)
                    ]
    return entry_clip
        

def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    sample_name, sample_label, sample_diff = [], [], []
    sample_rgb,sample_judge = [], []
    f1 = open(data_path,"r")
    for line in f1.readlines():
        filename,score,diff = line.split()

        comp_id, clip_id = filename.split("/")[-2:]
        clip_id = clip_id.split(".")[0]
        sample_rgb.append(process_rgb(comp_id,clip_id))
        judge_score = np.array(simple_data[(comp_id,int(clip_id))]['judge_scores'])
        sample_judge.append(judge_score)
        score = float(score) 
        diff = float(diff) 

        sample_name.append(filename)
        sample_label.append(score)
        sample_diff.append(diff)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    
    with open('{}/{}_diff.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_diff)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    error_path = "../error_txt.txt"
    error_f = open(error_path,"w")

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        
        data = read_xyz(
                    s, max_body=max_body, num_joint=num_joint)
        
        fp[i, :, 0:data.shape[1], :, :] = data
        # except
        #     error_f.write(s+"\n")

    print(f"len(sample_rgb):{len(sample_rgb)}")
    print(f"len(sample_judge):{len(sample_judge)}")
    np.save('{}/{}_rgb.npy'.format(out_path, part), np.array(sample_rgb,dtype=object))
    np.save('{}/{}_judge.npy'.format(out_path, part), np.array(sample_judge,dtype=object))

    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FineDiving Data Converter.')
    parser.add_argument(
        '--data_path', default='data/FineDiving')
    parser.add_argument(
        '--ignored_sample_path',
        default='resource/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/FineDiving')

    # part = ['val']
    part = ['test','train']
    arg = parser.parse_args()

    
    for p in part:
        out_path = os.path.join(arg.out_folder)
        data_path = arg.data_path + "/" +p + ".txt"
        # print(data_path)
        # st-gcn/data/FineDiving/train.txt
        gendata(data_path,
                out_path,
                None,
                benchmark="FineDiving",
                part=p)
