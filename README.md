# Gender detection (from scratch) using deep learning with keras and cvlib
The keras model is created by training SmallerVGGNet from scratch on around 2200 face images (~1100 for each class). Face region is cropped by applying face detection using cvlib on the images gathered from Google Images. It acheived around 96% training accuracy and ~90% validation accuracy. (20% of the dataset is used for validation)

# Python packages
1. numpy
2. opencv-python
3. tensorflow
4. keras
5. cvlib
## Install the required packages by executing the following command.

***$ pip install -r requirements.txt***

Note: Python 2.x is not supported

**Using Python==3.6 virtual environment is highly recommended.**

# Training 
You can download the dataset I gathered from Google Images from this link and train the network from scratch on your own if you are interested. You can add more images and play with the hyper parameters to experiment different ideas.
