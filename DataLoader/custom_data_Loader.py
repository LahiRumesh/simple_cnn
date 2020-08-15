import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import configparser

config = configparser.ConfigParser()
config.read('../user_inputs.ini')

parser=argparse.ArgumentParser()
parser.add_argument("--DATA_FOLDER", type=str, default=config.get('DataLoader','DATA_FOLDER'), help="Training Images folder")
parser.add_argument("--IMAGE_SIZE", type=str, default=config.getint('DataLoader','IMAGE_SIZE'), help="Image Size")
parser.add_argument("--TEST_SIZE", type=float, default=config.getfloat('DataLoader','TEST_SIZE'), help="Testing data percentage")
parser.add_argument("--SAVE_DATA", type=bool, default=config.getboolean('DataLoader','SAVE_DATA'), help="Save data to numpy array")

args = parser.parse_args()

    
def generate_data(DATA_FOLDER,IMAGE_SIZE,TEST_SIZE):

    classes=os.listdir(DATA_FOLDER)
    full_path=list(os.path.join(DATA_FOLDER,i) for i in classes)
    with open('classes.txt', 'w') as f:
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

    return training_data,testing_data


training_data,testing_data=generate_data(args.DATA_FOLDER,args.IMAGE_SIZE,args.TEST_SIZE)
    
if args.SAVE_DATA: #store data in a numpy array
    np.save("training_data.npy",training_data) #Save training data to numpy array
    np.save("testing_data.npy",testing_data) #Save testing data to numpy array

#if LOAD_DATA:
#    training_data=np.load("training_data.npy",allow_pickle=True)
#    testing_data=np.load("testing_data.npy",allow_pickle=True)



class Classification_DATASET(Dataset):
    def __init__(self,data_set,transform=None):
        self.data_set=data_set
        self.transform=transform
        #self.labels=labels

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


train_data = Classification_DATASET(training_data)
test_data = Classification_DATASET(testing_data)

train_loader = DataLoader(train_data, batch_size=2, shuffle=False)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True)
