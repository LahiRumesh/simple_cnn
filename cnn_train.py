from statistics import mode
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import torch.onnx as torch_onnx
from torch.autograd import Variable
from config import cfg
import wandb
from tqdm import tqdm
from time import sleep
from log_data import model_configs, wandb_logs, get_accuracy
from model_training.model_architecture import ModelArchitecture
cudnn.benchmark = True
wandb.login() #login to wandb account




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

        with open(os.path.join(self.checkpoints_dir,'classes.txt'), 'w') as f:
            for i,data in enumerate(self.class_names):
                f.write("%s\n" % data)

    def train_model(self,model, criterion,
                        optimizer, scheduler, num_epochs=25,
                        batch_size=4,
                        shuffle=True,num_workers=4):


        dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
                            for x in ['train', 'val']}

        device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        inputs, classes = next(iter(dataloaders['train']))
        out = torchvision.utils.make_grid(inputs)
        
        # Loard pre-trained model
        model_ = ModelArchitecture(model=cfg.model)
        model = model_.getModel(model,len(self.class_names))
        model = model.to(device)

        # wandb initialization
        wandb.init(project=f"simple_cnn-{os.path.basename(self.image_dir)}".replace("/", "-"), config=cfg)
        images = wandb.Image(out, caption=f"Sample_Batch-{os.path.basename(self.image_dir)}")
        wandb.watch(model, criterion, log="all", log_freq=10)
        wandb.log({"images": images
                       })
        
        model_configs(cfg=cfg)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # training mode
                else:
                    model.eval()   # evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    sleep(0.01)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics and wandb logs
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                get_accuracy(phase,epoch_loss,epoch_acc)
                wandb_logs(phase,epoch_loss,epoch_acc) # wandb log loss and accuracy
                

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()
        
        try:
            print(f'Best Val Accuracy: {best_acc:4f}')
        
        except KeyboardInterrupt:
            print(f'Best Val Accuracy: {best_acc:4f} Training Interrupted... Exporting ONNX model')
        

        # load best model weights and return for export
        model.load_state_dict(best_model_wts)
        return model



    def onnx_export(self,model,img_size=224,c_in=3,):

        input_shape = (c_in, img_size, img_size)
        model_prefix = os.path.basename(self.image_dir) + "_" + cfg.model
        counter = 1
    
        while os.path.exists(os.path.join(self.checkpoints_dir, f'{model_prefix + "_exp_" + str(counter)}.onnx')):
            counter += 1
        
        dummy_input = Variable(torch.randn(1, *input_shape,device="cuda"))
        torch_onnx.export(model, 
                          dummy_input, 
                          os.path.join(self.checkpoints_dir,f'{model_prefix + "_exp_" + str(counter)}.onnx'), 
                          verbose=False)
        print(f"{model_prefix}_exp_" + str(counter)+ ".onnx model successfully exported !")


if __name__ == '__main__':


    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((cfg.image_size,cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((cfg.image_size,cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_CNN = CNN_Trainer(cfg.data_dir,data_transforms)
    
    # model initilization
    model = getattr(models, cfg.model)(pretrained=cfg.pretrained)
    loss_criterion = getattr(nn, cfg.loss_criterion)()
    optimizer = getattr(optim, cfg.optimizer)(model.parameters(),
                                             lr=cfg.learning_rate, momentum=cfg.momentum, 
                                             weight_decay=cfg.weight_decay)
   
    # Decay Learning Rate
    exp_lr_scheduler = getattr(lr_scheduler, cfg.lr_scheduler)(optimizer, step_size=cfg.steps, gamma=cfg.gamma)

    cnn_model = train_CNN.train_model(model,loss_criterion,optimizer,exp_lr_scheduler,num_epochs=cfg.epochs)

    # export the ONNX model
    train_CNN.onnx_export(cnn_model,img_size=cfg.image_size)
