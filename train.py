# Import library packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
from torch import optim
import json
import argparse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

# Defining the arguments for the script
parser = argparse.ArgumentParser(description = 'Training script parser')

parser.add_argument('data_dir', help = 'Data directory', type = str)
parser.add_argument('--save_dir', help = 'Checkpoint save directory', type = str)
parser.add_argument('--arch', help='Architecture', type = str)
parser.add_argument('--learning_rate', help='Learning rate', type = float)
parser.add_argument('--hidden_units', help='Hidden units', type = int)
parser.add_argument('--epoch', help='epoch', type=int)
parser.add_argument('--gpu', help = 'Using the GPU', type = str)
parser.add_argument('--print_every', help = 'print every n-epochs', type = int)


# Setting the values to data loading
args = parser.parse_args()

# Data directory
data_dir = args.data_dir 

# Train directory
train_dir = data_dir + '/train'


# Validation directory
valid_dir = data_dir + '/valid'


# Test directory
test_dir = data_dir + '/test'



#Defining the device for using either cuda or cpu
device = ('cuda' if args.gpu == 'GPU' else 'cpu')


# loading the data if we are certain we have our values for data_dir

if data_dir:
    
    # Defining the transforms for the training, validation, and testing sets.
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transform)


    # Using the image datasets and the trainforms to define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle=True)

# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

output_layer = len(cat_to_name) # 102
    
def My_Model(arch, hidden_units):
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)

        # to freeze the parameters
        for param in model.parameters():
            param.requires_grad = False

        if hidden_units:

            # to change the classifier
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(model.classifier[0].in_features, hidden_units)),
                                                          ('relu1', nn.ReLU()),
                                                          ('dropout', nn.Dropout(p=0.2)),
                                                          ('fc2', nn.Linear(hidden_units, output_layer)),
                                                          ('output', nn.LogSoftmax(dim=1))]))



        else:

            # to change the classifier
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(model.classifier[0].in_features, 4096)),
                                                                               ('relu1', nn.ReLU ()),
                                                                                ('dropout1', nn.Dropout (p = 0.3)),
                                                                                ('fc2', nn.Linear (4096, 2048)),
                                                                                ('relu2', nn.ReLU ()),
                                                                                ('dropout2', nn.Dropout (p = 0.3)),
                                                                                ('fc3', nn.Linear (2048, 102)),
                                                                                ('output', nn.LogSoftmax (dim =1))
                                                                                ]))
  
           
                                 
    else:
        arch ='vgg16'
        model = models.vgg16(pretrained=True)

        # to freeze the parameters
        for param in model.parameters():
            param.requires_grad = False

        if hidden_units:

            # to change the classifier
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(model.classifier[0].in_features, hidden_units)),
                                                          ('relu1', nn.ReLU()),
                                                          ('dropout', nn.Dropout(p=0.2)),
                                                          ('fc2', nn.Linear(hidden_units, output_layer)),
                                                          ('output', nn.LogSoftmax(dim=1))]))


        else:

            # to change the classifier
            classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(model.classifier[0].in_features, 4096)),
                                                    ('relu1', nn.ReLU ()),
                                                    ('dropout1', nn.Dropout (p = 0.3)),
                                                    ('fc2', nn.Linear (4096, 2048)),
                                                    ('relu2', nn.ReLU ()),
                                                    ('dropout2', nn.Dropout (p = 0.3)),
                                                    ('fc3', nn.Linear (2048, 102)),
                                                    ('output', nn.LogSoftmax (dim =1))
                                                    ]))

    model.classifier = classifier
    return model, arch
    
    
def validation(new_model, validloader,criterion):
    
    new_model.to(device)
    
    valid_loss = 0
    valid_accuracy = 0
    
    for val_image, val_label in validloader:
        val_image, val_label = val_image.to(device), val_label.to(device)
        
        log_ps = new_model.forward(val_image)
        valid_loss += criterion(log_ps, val_label).item()
        
        ps = torch.exp(log_ps)
        
        equality = (val_label.data == ps.max(dim=1)[1])
        valid_accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, valid_accuracy
    
    
# Assigning My_Model() to a new variable called new_model                                                   
new_model, arch = My_Model(args.arch, args.hidden_units)
                                                           
                            
# Defining the Criterion and optimizer

criterion = nn.NLLLoss()

if args.learning_rate :
    optimizer = optim.Adam(new_model.classifier.parameters(), lr = args.learning_rate)
                                                       
else:
    optimizer = optim.Adam(new_model.classifier.parameters(), lr = 0.001)
                                                           
                                                            

# Training the network

new_model.to(device) 

if args.epoch:
    epoch = args.epoch
else:
    epoch = 5

steps = 0

if args.print_every:
    print_every = args.print_every
else:
    print_every = 5                                                   


for e in range(epoch):
    running_loss = 0#
    epoch_loss = 0


    print("Epoch {} out of {}...........................".format(e+1, epoch))

    for train_image,train_label in trainloader:
        steps +=1

        train_image, train_label = train_image.to(device), train_label.to(device)

        optimizer.zero_grad()
        output = new_model.forward(train_image)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        
        if steps % print_every == 0:
            new_model.eval()

            with torch.no_grad():
                valid_loss, valid_accuracy = validation(new_model, validloader, criterion)

            print('Training Loss: {:.3f} '.format(running_loss/print_every),
                  'Validation Loss:{:.3f} '.format(valid_loss/len(validloader)),
                  'Validation Accuracy:{:.3f}%'.format((valid_accuracy/len(validloader))*100))

            running_loss = 0   
            

            new_model.train()   
    print('epoch loss-----{:.3f}%'.format((epoch_loss/len(trainloader))*100))
    print()

               

# Save the checkpoint 

new_model.to('cpu')

new_model.class_to_idx = train_dataset.class_to_idx

model_checkpoint = {'arch': arch,
                    'classifier':new_model.classifier,
                    'model_state_dict': new_model.state_dict(),
                   'class_to_idx': new_model.class_to_idx,
                   'optim_state_dict': optimizer.state_dict()
                   }
                                                           
# Save directory

if args.save_dir:
    torch.save(model_checkpoint, args.save_dir + '/checkpoint_lates.pth')
else:
	torch.save(model_checkpoint, 'checkpoint_lates.pth')