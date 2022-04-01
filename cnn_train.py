import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.onnx as torch_onnx
from torch.autograd import Variable

cudnn.benchmark = True
plt.ion()   # interactive mode



class CNN_Trainer():

    def __init__(self,image_dir,
                      data_transforms) -> None:

        self.image_dir = image_dir
        self.data_transforms = data_transforms
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.image_dir, x),
                                          self.data_transforms[x])
                  for x in ['train', 'val']}

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        self.checkpoints_dir = os.path.join("models", os.path.basename(self.image_dir))
        try:
            os.makedirs(self.checkpoints_dir, exist_ok=True)
        except OSError:
            pass

    def train_model(self,model, criterion,
                        optimizer, scheduler, num_epochs=25,
                        batch_size=4,
                        shuffle=True,num_workers=4):

        dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
                            for x in ['train', 'val']}

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, classes = next(iter(dataloaders['train']))
        out = torchvision.utils.make_grid(inputs)
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.class_names))
        model = model.to(device)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model



    def onnx_export(self,model,img_size=224):

        model_prefix = os.path.basename(self.image_dir)
        onnx_model = os.path.join(self.checkpoints_dir, f'{model_prefix}.onnx')
        input_shape = (3, img_size, img_size)
        dummy_input = Variable(torch.randn(1, *input_shape,device="cuda"))
        torch_onnx.export(model, 
                          dummy_input, 
                          onnx_model, 
                          verbose=False)
        print("onnx model successfully exported !")


if __name__ == '__main__':


    data_dir = 'data'
    img_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    train_CNN = CNN_Trainer(data_dir,data_transforms)
    
    # model initilization
    
    model_ft = models.resnet50(pretrained=True)
    criterion = nn.CrossEntropyLoss()

    # Optimized parameters with LR
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    cnn_model = train_CNN.train_model(model_ft,criterion,optimizer_ft,exp_lr_scheduler,num_epochs=100)

    # export the ONNX model after training
    train_CNN.onnx_export(cnn_model,img_size=img_size)
