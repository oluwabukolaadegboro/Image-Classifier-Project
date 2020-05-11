# Image-Classifier-Project
**Developing an Image Classifier with Deep Learning.**

In this project, my knowledge of deep learning architectures was put to use in developing an image classifier using pytorch. The image classifier is in two parts. First it was built and trained on a deep neural network of a flower dataset which contains 102 categories of flowers. Then it was converted into a command line application. The application is a pair of Python scripts (train.py and predict.py) which runs from the command line. 

Transfer learning was used during the implementation of this project. The pretrained model, Vgg16 loaded from torchvision.models served as the pre-trained network while its pretrained layer parameters were frozen. Its parameters were modified to suit my desired model. To do this, I used the ReLU activation function and Dropout with p=0.2. 

The command line applications (train.py and predict.py) both had different role to play. For  example, train.py was used to train a new network on a dataset and saves the model as a checkpoint. On the other hand, predict.py uses the trained network to predict the class for an input image.

The image classifier was tested and proven to recognize a wide range of flower species. 


@Udacity Machine Learning Introduction Nanodegree
