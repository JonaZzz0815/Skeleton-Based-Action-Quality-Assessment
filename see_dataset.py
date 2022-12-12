import pickle

path = 'FineDiving_Dataset/Annotations/fine-grained_annotation_aqa.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)


print(data[("3mMenSpringboardFinal-EuropeanChampionships2021_2",1)])
