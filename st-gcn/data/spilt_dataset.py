import os
import pickle

import random

path = "../../FineDiving_Dataset/Annotations/FineDiving_fine-grained_annotation.pkl"  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

anno = open(path, 'rb')
anno_data = pickle.load(anno)
def get_verified_data(dataset_path,threshold=0.55):

    comp_col = os.listdir(dataset_path)
    verified_data = []
    for comp_name in comp_col:
        # skip Synchronised
        if "Synchronised" in comp_name:
            continue
        comp_name_path = os.path.join(dataset_path,comp_name)
        dir_col = os.listdir(comp_name_path)
        for dir_name in dir_col:
            dir_name_path = os.path.join(comp_name_path,dir_name)
            if "skip" in dir_name:
                continue
            full_len = len(anno_data[comp_name, 
                                    int(dir_name.split(".")[0])]['frames_labels'])
            f = open(dir_name_path,"r")
            cut_len = int(f.readlines()[0])
            # print(full_len,cut_len)
            if cut_len/full_len > threshold:
                verified_data.append([
                                    dir_name_path,
                                    float(anno_data[comp_name,int(dir_name.split(".")[0])]['dive_score']),
                                    float(anno_data[comp_name,int(dir_name.split(".")[0])]['difficulty'])
                                    # float(anno_data[comp_name,int(dir_name.split(".")[0])]['action_type'])
                                    ])
    return verified_data

def write_to_txt(out_path,data):
    f = open(out_path,"w")
    for item in data:
        to_write = f"{item[0][3:]} {str(item[1])} {str(item[2])}\n"
        f.write(to_write)
    f.close()

def split_train_test_dataset(dir_path,verified_data,ratio):
    random.shuffle(verified_data)

    full_len = len(verified_data)
    train_col = verified_data[:int(full_len*(ratio))]
    test_col = verified_data[int(full_len*ratio): ]
    write_to_txt(os.path.join(dir_path,"train.txt"),
                    train_col)
   
    write_to_txt(os.path.join(dir_path,"test.txt"),
                    test_col)

if __name__ == "__main__":
    dataset_path = "../../FineDiving_Skeleton_3"
    out_path = "FineDiving"
    threshold = 0.55
    verified_data = get_verified_data(dataset_path,threshold)
    print(f"length of seq > {threshold} is {len(verified_data)}!")
    
    split_train_test_dataset(out_path,verified_data,0.8)
    print("Done!")