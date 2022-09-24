import os
import onnx
import numpy as np
import onnxruntime as rt
import cv2
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt 

class CalAccuracy():

    def __init__(self,model_path,img_size,classes_path) -> None:

        self.model_path = model_path
        self.img_size = img_size
        self.classes_path = classes_path
        model = onnx.load(model_path)
        self.session = rt.InferenceSession(model.SerializeToString())


    def getImageList(self,
                        dirName,
                        endings=['.jpg','.jpeg','.png','.JPG']):

        listOfFile = os.listdir(dirName)
        allFiles = list()

        for i,ending in enumerate(endings):
            if ending[0]!='.':
                endings[i] = '.'+ending
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getImageList(fullPath,endings)
            else:
                for ending in endings:
                    if entry.endswith(ending):
                        allFiles.append(fullPath)               
        return allFiles#, os.path.basename(dirName)  



    def preprocess(self, img, use_transform=False):
                
        '''
        Image Pre-processing steps for inference
        img : image
        use_transform : use trasfromation step
        '''
        img = img / 255.
        img = cv2.resize(img, (self.img_size, self.img_size))        
        if use_transform:
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        else:
            h, w = img.shape[0], img.shape[1]
            y0 = (h - 224) // 2
            x0 = (w - 224) // 2
            img = img[y0 : y0+224, x0 : x0+224, :]

        img = np.transpose(img, axes=[2,0, 1])
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    def get_classes(self):

        '''
        Read the class file and return class names as an array
        '''
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def predict(self, dir_path,
                    use_transform=False,
                    test_dir='test_results'):

        '''
        Infer from the Test data set and visualize results
        '''
        
        counter = 1
        while os.path.exists(os.path.join(test_dir, f'{"test_exp_" + str(counter)}')):
            counter += 1

        try:
            os.makedirs(os.path.join(test_dir, f'{"test_exp_" + str(counter)}'), exist_ok=True)
        except OSError:
            pass

        df = pd.DataFrame()
        img_name, true_class, predict_class = [], [], []
        input_paths = self.getImageList(dir_path)
        for img_ in tqdm(input_paths):
            dir_class = os.path.basename(os.path.dirname(img_))
            img = cv2.imread(img_)
            img = self.preprocess(img, use_transform=use_transform)
            inputs = {self.session.get_inputs()[0].name: img}
            preds = self.session.run(None, inputs)[0]
            preds = np.squeeze(preds)
            a = np.argsort(preds)[::-1]
            labels = self.get_classes()
            img_name.append(img_)
            true_class.append(dir_class)
            predict_class.append(labels[a[0]])

        df["Image"] = img_name
        df["True_Class"] = true_class
        df["Predicted_Class"] = predict_class
        df.to_csv(os.path.join(os.path.join(test_dir, f'{"test_exp_" + str(counter)}'),"predict_results.csv"))

        df['result'] = df['True_Class'] == df['Predicted_Class']
        mean_accuracy = df.value_counts('result')[True] / df['result'].count()
        df2 = pd.concat([df.groupby('True_Class')['result'].sum().reset_index(),df.value_counts('True_Class').reset_index()],axis=1,join='inner')
        df2['Accuracy'] = df2.iloc[:,1]/df2.iloc[:,3]
        df2 = df2.loc[:, ~df2.columns.duplicated()]
        df2 = df2.drop(columns=df2.columns.values[1:3])
        df2.to_csv(os.path.join(os.path.join(test_dir, f'{"test_exp_" + str(counter)}'),"accuracy_results.csv"))
        print('Model Accuracy : %s' %(mean_accuracy))
        df2.plot(x="True_Class", y='Accuracy', kind='bar') 
        plt.savefig(os.path.join(os.path.join(test_dir, f'{"test_exp_" + str(counter)}'),"test_accuracy.png"))
        plt.show()
        




if __name__ == '__main__':

    '''
    python test_accuracy.py --model_path=models/cats_vs_dogs/cats_vs_dogs_resnet18_exp_1.onnx --class_path=models/cats_vs_dogs/classes.txt --img_dir=Classification/Data/cats_vs_dogs/test --image_size=224 --use_transform=True
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/cats_vs_dogs/cats_vs_dogs_resnet50_exp_8.onnx', help='ONNX model path')
    parser.add_argument('--class_path', type=str, default='models/cats_vs_dogs/classes.txt', help='Class file path which contain class names')
    parser.add_argument('--img_dir', type=str, default='Classification/Data/cats_vs_dogs/test', help='Test Images Dir path')
    parser.add_argument('--image_size', type=int, default=224, help='Input Image size (Used for the training)')
    parser.add_argument('--use_transform', type=bool, default=True, help='Use image transforms in pre-processing step')
    args = parser.parse_args()

    cal_accuracy = CalAccuracy(args.model_path,args.image_size,args.class_path)
    cal_accuracy.predict(args.img_dir,use_transform=args.use_transform)

