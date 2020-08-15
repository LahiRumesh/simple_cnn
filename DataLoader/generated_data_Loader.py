import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
import torch
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
parser.add_argument("--SAVE_DATA_FOLDER", type=str, default=config.get('DataLoader','SAVE_DATA_FOLDER'), help="Save data to numpy array")
parser.add_argument("--CLASSES_FILE", type=str, default="classes.txt", help="output text file contain all classes")
parser.add_argument("--TESTING_DATA", type=str, default="training_data.npy", help="output numpy training data file name")
parser.add_argument("--TRAINING_DATA", type=str, default="testing_data.npy", help="output numpy testing data file name")
args = parser.parse_args()

training_data=np.load(os.path.join(args.SAVE_DATA_FOLDER,args.TRAINING_DATA),allow_pickle=True)
testing_data=np.load(os.path.join(args.SAVE_DATA_FOLDER,args.TESTING_DATA),allow_pickle=True)

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

print(train_data)

train_loader = DataLoader(train_data, batch_size=2, shuffle=False)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True)
