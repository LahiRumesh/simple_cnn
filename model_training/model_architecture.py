import torch.nn as nn
from tabulate import tabulate

class ModelArchitecture():
    def __init__(self, model='resnet50') -> None:
        self.model = model
        self.resnet = ["resnet18","resnet34","resnet50","resnet101","resnet152"]
        self.vgg = ["vgg13","vgg13_bn","vgg16","vgg16_bn","vgg19","vgg19_bn"]
        self.alexnet = ["alexnet"]
        self.squeezenet = ["squeezenet1_0","squeezenet1_1"]
        self.densenet = ["densenet121", "densenet169", "densenet161" ,"densenet201"]
      

    def getModel(self,
                model,
                out_features):
        
        if self.model in self.resnet:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, out_features)

        elif self.model in self.vgg:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,out_features)

        elif self.model in self.alexnet:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,out_features)

        elif self.model in self.squeezenet:
            model.classifier[1] = nn.Conv2d(512, out_features, kernel_size=(1,1), stride=(1,1))

        elif self.model in self.densenet:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, out_features)

        else:
            
            print('Available Models')
            model_table = [["Resnet",["resnet18","resnet34","resnet50","resnet101","resnet152"]],
                          ["VGG",["vgg13","vgg13_bn","vgg16","vgg16_bn","vgg19","vgg19_bn"]],
                          ["Densenet",["densenet121", "densenet169", "densenet161" ,"densenet201]"]],
                          ["Squeezenet",["squeezenet1_0","squeezenet1_1"]],
                          ["Alexnet", ["alexnet"]]]
            headers = ["Architectures", "Available Models"]
            print(tabulate(model_table,headers,tablefmt="psql"))
            raise Exception("Invalid Model Architecture !! Please use available models")
     
        
        return model