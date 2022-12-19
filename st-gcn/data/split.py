
import os
import pickle

import random

path = "../../FineDiving_Dataset/Annotations/FineDiving_fine-grained_annotation.pkl"  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

anno = open(path, 'rb')
anno_data = pickle.load(anno)

test_path = "../../FineDiving_Dataset/train_test_split/test_split.pkl"
test = open(test_path, 'rb')
test_data = pickle.load(test)

train_path = "../../FineDiving_Dataset/train_test_split/train_split.pkl"
train = open(train_path, 'rb')
train_data = pickle.load(train)


def get_non_syn_data(mode,path):
    if mode == "train":
        data = train_data
    else:
        data = test_data
    col_data = []
    for comp_id, clip_id in data:
        if "Synchronised" in comp_id:
            continue
        now_path = os.path.join(path,comp_id,str(clip_id)) + ".txt"
        col_data.append([
                        now_path,
                        float(anno_data[(comp_id,clip_id)]['dive_score']),
                        float(anno_data[(comp_id,clip_id)]['difficulty'])
        ])
    return col_data

def write_to_txt(out_path,data):
    f = open(out_path,"w")
    for item in data:
        to_write = f"{item[0][3:]} {str(item[1])} {str(item[2])}\n"
        f.write(to_write)
    f.close()

def split_train_test_dataset(dataset_path,out_path):
    test_one = get_non_syn_data("test",dataset_path)
    train_one = get_non_syn_data("train",dataset_path)
    
    print(f"Test len is {len(test_one)}")
    print(f"Train len is {len(train_one)}")

    write_to_txt(os.path.join(out_path,"train.txt"),
                    train_one)
   
    write_to_txt(os.path.join(out_path,"test.txt"),
                    test_one)



if __name__ == "__main__":
    dataset_path = "../../FineDiving_Skeleton_3"
    out_path = "FineDiving"
    
    split_train_test_dataset(dataset_path,out_path)
    print("Done!")