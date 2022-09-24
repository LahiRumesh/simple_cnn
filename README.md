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

    cfg.pretrained = True  # use pretrained weights for training

    #Early Stopping
    cfg.use_early_stopping = True # use Early stopping
    cfg.patience = 8 # how many epochs to wait before stopping when accuracy is not improving
    cfg.min_delta = 0.0001 # minimum difference between new accuracy and old accuracy for new accuracy to be considered as an improvement                   
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


Run **cnn_train.py** to start the training, all the logs will save in [wandb](https://wandb.ai/site), and ONNX weight files will save in the "**_models/Image_Folder_**" folder for each training experiment with the model name. 

---

### Inference

- After the training process, use the exported ONNX model for inference using **cnn_inference.py**

```bash
python cnn_inference.py --model_path=models/ImageFolder/ImageFolder_resnet18_exp_1.onnx --class_path=models/ImageFolder/classes.txt --img_path=test1.jpg --image_size=224 --use_transform=True
```



 ```bash
 '''
  Args:
 '''
    --model_path :  ONNX model path
    --class_path : Class file (classes.txt) path contain class names
    --img_path  : Input image path
    --image_size : input image size
    --show_image : Display the image
    --use_transform : Use image transforms in pre-processing step (During the training, process images are Normalize with a mean and standard deviation)                 
```

---

### Calculate Test Accuracy

- Use the **test_accuracy.py** to calculate the ONNX model accuracy on the test data.

```bash
python test_accuracy.py --model_path=models/ImageFolder/ImageFolder_resnet18_exp_1.onnx --class_path=models/ImageFolder/classes.txt --img_dir=Image_Folder/test --image_size=224 --use_transform=True
```
The following illustrates 3 classes of the test image folder

```bash
├── Image_Folder
     ├── test
        │───── class1
        │     ├── class1.0.jpg
        │     ├── class1.1.jpg
        │     ├── class1.2.jpg
        │     ├── .........
        │     └── class1.500.jpg
        │
        │───── class2
        │     ├── class2.0.jpg
        │     ├── class2.1.jpg
        │     ├── class2.2.jpg
        │     ├── .........
        │     └── class2.500.jpg
        │
        └───── class3
               ├── class3.0.jpg
               ├── class3.1.jpg
               ├── class3.2.jpg
               ├── .........
               └── class3.500.jpg   
     
```

All the test results will save in the folder "**_test_results_**" folder for each test experiment.

 ```bash
 '''
  Args:
 '''
    --model_path :  ONNX model path
    --class_path : Class file (classes.txt) path contain class names
    --img_dir  : Test images folder path
    --image_size : input image size
    --use_transform : Use image transforms in pre-processing step (During the training, process images are Normalize with a mean and standard deviation)                 
```

---

### Reference:

- [PyTorch TRANSFER LEARNING FOR COMPUTER VISION](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [PyTorch MODELS AND PRE-TRAINED WEIGHTS](https://pytorch.org/vision/stable/models.html)
- [ONNX Model Zoo](https://github.com/onnx/models)