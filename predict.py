# import library packagess
import torch
import torch as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
from torch import optim
import json
import argparse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


# Defining the Arguments for the script
parser = argparse.ArgumentParser(description = 'Training script parser')

parser.add_argument('image_dir', help = 'Image directory', type = str)
parser.add_argument('load_dir', help = 'Loading the saved model', type = str)
parser.add_argument('--top_k', help = 'Top K most likely classes', type = int)
parser.add_argument('--category_names', help = 'Mapping of categories to real names', type = str)
parser.add_argument('--gpu', help = 'Using the GPU', type = str)


# A function to load the checkpoint and rebuild the model

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)

    if checkpoint ['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        model = models.vgg16(pretrained = True)
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False # turning off the grad when testing.
        
    return model


# To process a PIL image for use in a PyTorch model

def process_image(image):
    print('debug', image)
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image)
    
    mean = [0.485, 0.456, 0.406]
    
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])
    
    transform_img = transform(img)
    
    np_transform_img = np.array(transform_img)
    
    transposed_np_transform_img = np_transform_img.transpose((0,1,2))
    
    return transposed_np_transform_img



# To implement the code to predict the class from an image file

def predict(image_path, model, topk, device):
    ''' Predict    the class (or classes) of an image using a trained deep learning model.'''
    
    image = process_image(image_path)
    if device == 'cuda':
        im = torch.from_numpy (image).type (torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy (image).type (torch.FloatTensor)

    im = im.unsqueeze(dim=0)

    model.to(device)
    im.to(device)

    with torch.no_grad():
        log_ps = model.forward(im)
        ps = torch.exp(log_ps)

        top_probs, prob_indices = ps.topk(topk)

        top_probs = top_probs.cpu().numpy()
        prob_indices = prob_indices.cpu().numpy()

        top_probs = top_probs.tolist() [0] # it takes the first value of the top_prob
        prob_indices = prob_indices.tolist() [0]  # it takes the first value of the top_class


        # To convert from the indices to the actual labels, we need to get the class names from the dictionary.
        index_to_actual_class_labels = {val:key for key, val in model.class_to_idx.items()}

        top_classes = [index_to_actual_class_labels[each] for each in prob_indices]

        return(top_probs, top_classes)

    
# Setting values for data loading
args = parser.parse_args ()
file_path = args.image_dir


# Define GPU/cpu
if args.gpu == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
    
    
# Loading the JSON file if provided, else load the default file name
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass
    
# Loading the model from the checkpoint provided
model = load_checkpoint(args.load_dir)


# Defining the number of classes to be predicted. Default = 1.
if args.top_k:
    top_classes = args.top_k
else:
    top_classes = 1


    
# To display an image along with the top 5 classes

def show_image(image_path, model):
    
    fig, ax = plt.subplots()
    
    image = process_image(image_path)
    
    #calculate probabilities and classes
    top_probs, top_class = predict(image_path, model, top_classes, device)
    
    #preparing class_names using mapping with cat_to_name
    class_names = [cat_to_name [item] for item in top_classtop_classes]
    
    ax.barh(class_names, top_probs, align='center')
    
    ax.set_yticklabels(class_names)
    ax.set_ylabel('Predicted Species of Flower')
    ax.set_xlabel('Probability of the predicted species')
    
    imshow(image) # displays the image
    
    plt.show # displays the graph    
    
    
# Calculating the probabilities and classes.
probs, classes = predict (file_path, model, top_classes, device)

# Preparing class_names using mapping with cat_to_name
class_names = [cat_to_name [item] for item in classes]

for l in range (top_classes):
     print("Number: {}/{}.. ".format(l+1, top_classes),
            "Class name: {}.. ".format(class_names [l]),
            "Probability: {:.3f}..% ".format(probs [l]*100),
            )
   