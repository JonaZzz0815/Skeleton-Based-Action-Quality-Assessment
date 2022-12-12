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

path = '../FineDiving_Dataset/Annotations/fine-grained_annotation_aqa.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

anno = open(path, 'rb')
anno_data = pickle.load(anno)


def get_score(comp_id,clip_id):
    return anno_data[(comp_id,clip_id)]


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    sample_name, sample_label = [], []
    f1 = open(data_path,"r")
    for line in f1.readlines():
        filename,score = line.split()
        score = float(score) 
        sample_name.append(filename)
        sample_label.append(score)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
                    s, max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FineDiving Data Converter.')
    parser.add_argument(
        '--data_path', default='data/FineDiving')
    parser.add_argument(
        '--ignored_sample_path',
        default='resource/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/FineDiving')

    # part = ['train']
    part = [ 'val', 'test','train']
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
