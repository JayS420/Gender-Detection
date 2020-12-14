# Gender detection (from scratch) using deep learning with keras and cvlib
The keras model is created by training from scratch on around 2000 face images (~1000 for each class). Face region is cropped by applying face detection using cvlib on the images gathered from Google Images. It acheived around 99% training accuracy and ~95% validation accuracy. (20% of the dataset is used for validation).

# Python packages
1. numpy
2. opencv-python
3. tensorflow
4. keras
5. cvlib
6. scikit-learn
7. matplotlib
## Install the required packages by executing the following command.

***$ pip install -r requirements.txt***

Note: Python 2.x is not supported

**Using Python==3.6 virtual environment is highly recommended.**

# Training 
You can download the dataset I gathered from Google Images and train the network from scratch on your own if you are interested. You can add more images and play with the hyper parameters to experiment different ideas.

Depending on the hardware configuration of your system, the execution time will vary. On CPU, training will be slow. After the training, the model file will be saved in the current path as gender_detection.model.

If you have an Nvidia GPU, then you can install tensorflow-gpu package. It will make things run a lot faster
# Help
If you are facing any difficulty, feel free to create a new issue or reach out on linkdin - https://www.linkedin.com/in/jay-singh-318290199/
