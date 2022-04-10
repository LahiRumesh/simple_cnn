import onnx
import numpy as np
import onnxruntime as rt
import cv2


class CNNInference():
    def __init__(self,model_path,img_size,classes_path) -> None:

        self.model_path = model_path
        self.img_size = img_size
        self.classes_path = classes_path
        model = onnx.load(model_path)
        self.session = rt.InferenceSession(model.SerializeToString())

    def get_image(self, path, show=False):
        img = cv2.imread(path)
        if show:
            cv2.imshow("Frame",img)
            cv2.waitKey(0)
        return img


    def preprocess(self, img, use_transform=False):
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
        args : class file (classes.txt)
        '''
        with open(self.classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def predict(self, path, 
                    show_img=False, 
                    use_transform=False):
        img = self.get_image(path, show=show_img)
        img = self.preprocess(img, use_transform=use_transform)
        inputs = {self.session.get_inputs()[0].name: img}
        preds = self.session.run(None, inputs)[0]
        preds = np.squeeze(preds)
        a = np.argsort(preds)[::-1]
        labels = self.get_classes()
        print('Predicted Class : %s' %(labels[a[0]]))




if __name__ == '__main__':


    model_path = '/home/lahiru/Computer_Vision/Classification/simple_cnn/models/cat_and_dogs/cat_and_dogs_resnet18_exp_1.onnx'
    class_path = '/home/lahiru/Computer_Vision/Classification/simple_cnn/models/cat_and_dogs/classes.txt'
    cnn_infer = CNNInference(model_path,224,class_path)
    img_path = '/home/lahiru/Computer_Vision/Classification/Data/cat_and_dogs/val/dogs/dog.4033.jpg'
    cnn_infer.predict(img_path)

