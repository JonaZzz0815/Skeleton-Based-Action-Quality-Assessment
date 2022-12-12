import os
import pickle
import matplotlib.pyplot as plt

path = 'FineDiving_Dataset/Annotations/fine-grained_annotation_aqa.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

# print(data[("3mMenSpringboardFinal-EuropeanChampionships2021_2",1)])

def check_length(file_name,dir_name,item):
    full_len = len(data[(dir_name,int(item))][-1])
    f = open(file_name,"r")
    
    try:
        cut_len = int(f.readlines()[0])
        return full_len,cut_len
    except:
        return file_name



if __name__ == "__main__":
    sk_path = "FineDiving_Skeleton"
    ratio = []
    empty_txt = []
    for dir_name in os.listdir(sk_path):
        dir_path = os.path.join(sk_path, dir_name)
        for item in os.listdir(dir_path):
            txt_path = os.path.join(dir_path, item)
            ans = check_length(txt_path,dir_name,item.split(".")[0])
            if type(ans) == str:
                empty_txt.append(ans)
            else:
                ratio.append(float(ans[1]/ans[0]))
    plt.hist(ratio)
    print(len(ratio))
    print(len(empty_txt))
    plt.savefig("hist.jpg")
    f = open("empty_txt.txt","w")
    f.write("\n".join(empty_txt))
    f.close()
    # print(empty_txt,file="empty_txt.txt")
    
    # full,cut = check_length("FineDiving_Skeleton/3mMenSpringboardFinal-EuropeanChampionships2021_1/0.txt",
    #             "3mMenSpringboardFinal-EuropeanChampionships2021_1",
    #             "0")
    