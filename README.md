Gender detection (from scratch) using deep learning with keras and cvlib
The keras model is created by training SmallerVGGNet from scratch on around 2200 face images (~1100 for each class). Face region is cropped by applying face detection using cvlib on the images gathered from Google Images. It acheived around 96% training accuracy and ~90% validation accuracy. (20% of the dataset is used for validation)

Update :
Checkout the gender detection functionality implemented in cvlib which can be accessed through a single function call detect_gender().

Python packages
numpy
opencv-python
tensorflow
keras
requests
progressbar
cvlib
Install the required packages by executing the following command.

$ pip install -r requirements.txt

Note: Python 2.x is not supported

Make sure pip is linked to Python 3.x (pip -V will display this info).

If pip is linked to Python 2.7. Use pip3 instead. pip3 can be installed using the command sudo apt-get install python3-pip

Using Python virtual environment is highly recommended.

Usage
image input
$ python detect_gender.py -i <input_image>

webcam
$ python detect_gender_webcam.py

When you run the script for the first time, it will download the pre-trained model from this link and place it under pre-trained directory in the current path.

(If python command invokes default Python 2.7, use python3 instead)
