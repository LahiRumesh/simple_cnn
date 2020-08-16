import os
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms


def generate_data(DATA_FOLDER,IMAGE_SIZE=224,TEST_SIZE=0.1,SAVE_DATA=True,CLASSES_FILE="classes.txt",OUT_DATA="../generated_data"):

    if not os.path.exists(OUT_DATA):
        os.makedirs(OUT_DATA)

    classes=os.listdir(DATA_FOLDER)
    full_path=list(os.path.join(DATA_FOLDER,i) for i in classes)
    with open(os.path.join(OUT_DATA,CLASSES_FILE), 'w') as f:
        for i,data in enumerate(classes):
            f.write("%s\n" % data)

    data_list=[]
    labels = {full_path[x]:x for x in range(len(full_path))}
    for label in labels:
        for f in tqdm(os.listdir(label)):
            try:
                path=os.path.join(label,f)
                img=cv2.imread(path)
                img=cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
                data_list.append([np.array(img),labels[label]])

            except Exception as e:
                pass

    np.random.shuffle(data_list)
    test_size=int(len(data_list)*TEST_SIZE)
    training_data,testing_data=data_list[test_size:],data_list[:test_size]
    if SAVE_DATA:
        np.save(os.path.join(OUT_DATA,"training_data2.npy"),training_data)
        np.save(os.path.join(OUT_DATA,"testing_data2.npy"),testing_data)


    return training_data,testing_data


class Classification_DATASET(Dataset):
    def __init__(self,data_set,transform=None):
        self.data_set=data_set
        self.transform=transform

    def __len__(self):
        return(len(self.data_set))

    def __getitem__(self,idx):
        data = self.data_set[idx][0]
        data=data.astype(np.float32).reshape(3,100,100)

        if self.transform:
            data=self.transform(data)
            return (data,self.data_set[idx][1])
        else:
            return (data,self.data_set[idx][1])

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


training_data,testing_data=generate_data("../Data")


train_loader = DataLoader(Classification_DATASET(training_data), batch_size=2, shuffle=False)
test_loader = DataLoader(Classification_DATASET(testing_data), batch_size=2, shuffle=True)
