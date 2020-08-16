import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Simple_CNN_Trainer():
    def __init__(self,model,learning_rate,num_epochs,train_loader,test_loader):
        self.model=model
        self.learning_rate=learning_rate
        self.num_epochs=num_epochs
        self.train_loader=train_loader
        self.test_loader=test_loader

    
    
    
    
    
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

    model = model_ft.to(device)
    learning_rate = 0.1
    num_epochs = 5

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            print("Label Shape=",labels.shape)
            # Forward pass
            outputs = model(images)
            print("output Shape=",outputs.shape)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 2 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(2 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')