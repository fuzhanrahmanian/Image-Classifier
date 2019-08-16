# Imports here
#%matplotlib inline
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import json
import PIL
from PIL import Image
from torchvision import datasets, transforms, models
from torch import optim
from torch import nn
from collections import OrderedDict
from workspace_utils import active_session
from torch.autograd import Variable


def load_data(data_structure):
    
    #Directory for different data
    data_dir = data_structure
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Create Transforms
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
     
    validation_transform = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform= validation_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    #Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    name_of_classes = test_data.classes
    name_of_classes_valid = validation_data.classes


    
    return train_data, train_loader, validation_loader, test_loader, name_of_classes


def network(structure, hidden_layers, learning_rate, gpu, dropout, name_of_classes):
    
    arch = {"vgg16":25088,
            "densenet121":1024}
    #Load the mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    for i in name_of_classes:
        name = cat_to_name[i]
        print(name)
        
    #NN Structure
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        model
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.name = "densenet121"  
        model
    else:
        print("{} is not a valid model. Use vgg16,densenet121.".format(structure))
    
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(arch[structure], hidden_layers, bias = True)),
                              ('relu1', nn.ReLU()),
                              ('d_out1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_layers, 512, bias = True)),
                              ('relu2', nn.ReLU()),
                              ('d_out2', nn.Dropout(0.3)),
                              ('fc3', nn.Linear(512, 102, bias= True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    '''
    if torch.cuda.is_available() and gpu = "gpu":
        device = torch.device('cuda:0')
    else
        device = torch.device('cpu')
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    if torch.cuda.is_available():
        print("We have GPU")
    #and gpu == "gpu"
    device
    model.to(device)
    return model, criterion, optimizer, device

'''def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for ii, (inputs, labels) in enumerate(testloader):

        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1) 
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return test_loss, accuracy
        '''
    
def train_network(epochs, model, train_loader, optimizer, criterion, device, save_dir, validation_loader, test_loader, train_data):
    print_every = 40
    steps = 0

    print("Using {}".format(model))
    print("Training process initializing on {}\n".format(device))
    with active_session():
        for e in range(epochs):
            running_loss = 0
            model.train()

            for ii, (inputs, labels) in enumerate(train_loader):
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                #Foward propagation
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)

                #Backward
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    #validation_loss = 0
                    #validation_accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        test_loss = 0
                        accuracy = 0

                        for ii, (inputs, labels) in enumerate(validation_loader):

                            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim = 1) 
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        #validation_loss, accuracy = validation(model, validation_loader, criterion, device)


                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every),
                          "Validation Loss {:.4f}".format(test_loss/len(test_loader)),
                          "Accuracy: {:.4f}".format(accuracy/len(test_loader)))

                    running_loss = 0
                    model.train()
        print("\nTraining process is now complete!") 
        print("\nTesting the network")
        right_answer = 0
        total_images = 0

        # turn off gradiant and save memory and computation
        model.to('cuda:0')
        with torch.no_grad():
            model.eval()
        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            right_answer += (predicted==labels).sum().item()
        print('Accuracy on the test images {:.4f}'.format(100*right_answer/total_images))

        print('Saving the Network')
        model.class_to_idx = train_data.class_to_idx
        model.cpu
        torch.save({
                    'architecture':model.name,
                    'classifier':model.classifier,
                    'hidden_layer': 4096,
                    'state_dict':model.state_dict(),
                    'class_to_idx':model.class_to_idx},
                    save_dir)
        
        
def load_checkpoint(filepath, structure):
    
    checkpoint = torch.load(filepath)
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        model
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.name = "densenet121"  
        model

    #freeze 
    for param in model.parameters():
        param.requires_grad=False
        
    model.class_to_idx = checkpoint['class_to_idx']   
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image_path):
    pil_image = Image.open(image_path)
    
    change = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])])

    image_tensor = change(pil_image)           
    return image_tensor 


def predict(image_path, model, top_k):   
    
        model.to('cuda:0')
        
        #image = Image.open(image_path)
        
        image = process_image(image_path)
        
        
        # covert the image into the 1 dimentional vector 
        image = np.expand_dims(image, 0)
        # from numpy to torch 
        image = torch.from_numpy(image)
        model.eval()
        inputs =  Variable(image).to('cuda:0')
        # forward propagation 
        logps = model.forward(inputs)
        ps = F.softmax(logps, dim=1)
        # for the top 5 classes 
        topk = ps.cpu().topk(top_k)
        # return a python list and removing the single dimentional entries from the shape of array 
        return (e.data.numpy().squeeze().tolist() for e in topk)
    