# Simple CNN

#### ```Simple CNN``` is a pipeline which can be use to train and infer CNN models by use of PyTorch and ONNX. It's simple and easy to USE !!! 🔥🔥
___
### Install

- Clone the repo and install **requirements.txt** in a Python environment 

```bash
    git clone https://github.com/LahiRumesh/simple_cnn.git
    cd simple_cnn
    pip install -r requirements.txt
```
---

### Data Preparation

- Split images into **train** and **val** folders with each class the Image Folder 📂.. i.e for cat vs dogs classification, there should be a cat folder and dog folder in both train and val. The following folder structure illustrates 3 classes


```bash
├── Image_Folder
     ├── train
     │   │───── class1
     │   │     ├── class1.0.jpg
     │   │     ├── class1.1.jpg
     │   │     ├── class1.2.jpg
     │   │     ├── .........
     │   │     └── class1.500.jpg
     │   │
     │   │───── class2
     │   │     ├── class2.0.jpg
     │   │     ├── class2.1.jpg
     │   │     ├── class2.2.jpg
     │   │     ├── .........
     │   │     └── class2.500.jpg
     │   │
     │   └───── class3
     │          ├── class3.0.jpg
     │          ├── class3.1.jpg
     │          ├── class3.2.jpg
     │          ├── .........
     │          └── class3.500.jpg   
     │
     └── val
         │───── class1
         │     ├── class1.501.jpg
         │     ├── class1.502.jpg
         │     ├── class1.503.jpg
         │     ├── .........
         │     └── class1.600.jpg
         │
         │───── class2
         │     ├── class2.501.jpg
         │     ├── class2.502.jpg
         │     ├── class2.503.jpg
         │     ├── .........
         │     └── class2.600.jpg
         │
         └───── class3
               ├── class3.501.jpg
               ├── class3.502.jpg
               ├── class3.503.jpg
               ├── .........
               └── class3.600.jpg

```

---

### Training
  #### After the data preparation, it's time for the training !  
- Use the **config.py** to set the parameters, here are few parameters.
     
 ```bash
    cfg.data_dir = 'Data/Images/Image_Folder' # Image Folder path which contain train and val folders 
    cfg.device = '0' # cuda device, i.e. 0 or 0,1,2,3    
    cfg.image_size = 224 #input image size
    cfg.batch_size = 8 # batch size
    cfg.epochs = 50 #number of epochs

    cfg.model = 'resnet18' # torch vision classification model architectures for image classification 
                           # i.e. resnet18 or vgg16, alexnet, densenet121, squeezenet1_0

    cfg.pretrained = True  # use pretrained weights for training-                    
```


- Here are the Available pre-trained models in ```Simple CNN```

  <table border="3">
  <tr>
  <td><b>Architectures</td>
  <td><b>Available Models</td>
  </tr>
  <td>Resnet</td>
  <td>resnet18, resnet34, resnet50, resnet101, resnet152</td>
  <tr>
  <td>VGG</td>
  <td>vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn</td>
  <tr>
  <td>Densenet</td>
  <td>densenet121, densenet169, densenet161 , densenet201</td>
  <tr>
  <td>Squeezenet</td>
  <td>squeezenet1_0, squeezenet1_1</td>
  <tr>
  <td>Alexnet</td>
  <td>alexnet</td>
  </table


Run **cnn_train.py** to start the training, all the logs will be save in [wandb](https://wandb.ai/site), and ONNX weight files will save in the "**_models/Image_Folder_**" folder for each training experiments with the model name. 

---

### Inference

- After the training process, use the exported ONNX model for inference using **cnn_inference.py**

```bash
python cnn_inference.py --model_path=models/ImageFolder/ImageFolder_resnet18_exp_1.onnx --class_path=ImageFolder/ImageFolder/classes.txt --img_path=test1.jpg --image_size=224 
```

 ```bash
 '''
  Args:
 '''
    --model_path :  ONNX model path
    --class_path : Class file (classes.txt) path contain class names
    --img_path  : Input image path
    --image_size : input image size                    
```
---

### Reference:

- [PyTorch TRANSFER LEARNING FOR COMPUTER VISION](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [PyTorch MODELS AND PRE-TRAINED WEIGHTS](https://pytorch.org/vision/stable/models.html)
- [ONNX Model Zoo](https://github.com/onnx/models)