# Simple CNN

#### ```Simple CNN``` is a pipeline which can be use to train and infer CNN models by use of PyTorch and ONNX. It's simple and easy to USE !!! 🔥🔥
___
### Install

- Clone the repo and install **requirements.txt** in a Python environment 

```bash
    git clone git@github.com:LahiRumesh/simple_cnn.git
    cd simple_cnn
    pip install -r requirements.txt
```
---

### Data Preparation

- Split images into **train** and **val** folders in the Image Folder 📂

 * #### Image Folder
    * **train**
      * image1.jpg
      * image2.jpg
      * image3.jpg
      * ..............
    * **val**
      * image101.jpg
      * image102.jpg
      * image103.jpg
      * ..............

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
                           # i.e. resnet18 or vgg16, alexnet, googlenet, resnet50

    cfg.pretrained = True  # use pretrained weights for training-                    
```
- Please checkout for more [MODELS AND PRE-TRAINED WEIGHTS](https://pytorch.org/vision/stable/models.html) from PyTorch 

Run **cnn_train.py** to start the training, all the logs will be save in [wandb](https://wandb.ai/site), and ONNX weight files will save in the "**_models/Image_Folder_**" folder for each training experiments. 

