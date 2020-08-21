# Image-Classifier-Project
**Developing an Image Classifier with Deep Learning.**

This was my 2nd project as part of the Udacity Machine Learning Nanodegree Program. I used my knowledge of deep learning architectures to develop an Image Classifier using Pytorch deep learning framework. The project was in two parts. First, the deep neural network was built and trained on a flower dataset which contains over 100 categories. Then, it was converted into a command line application. The application is a pair of Python scripts (train.py and predict.py) which runs from the command line.

Transfer learning was used during the implementation of this project. The pretrained model, Vgg16 loaded from torchvision.models served as the pre-trained network while its pretrained layer parameters were frozen. During training, I modified the hyperparameters to suit my desired model. To this end, I did a bit of exploration before exploitation.

The command line applications (train.py and predict.py) both had different roles to play. For example, train.py was used to train a new network on a dataset and save the model as a checkpoint. On the other hand, predict.py uses the trained network to predict the class for an input image.

At the end of this project, the trained network was able to recognize a wide range of flower species.


![Flower Classification 1](/images/Flower_1.png)   

#

![Flower Classification 2](/images/Flower_2.png)
