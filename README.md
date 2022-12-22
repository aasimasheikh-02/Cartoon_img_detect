# Tom  and Jerry detection on Yolov5 using Jetson Nano
# Aim and Objectives
# Aim
To  create a model  which will detect object  based  on whether  it is Tom  or Jerry.  
# Objectives
The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting the object.
# Abstract
•	An object is classified based on whether it is  Tom or Jerry.

•	We have completed this project on jetson nano which is a very small computational device.

•	A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

•	One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.
# Introduction
•	This  project is based on a detection model which detect whether the object is Tom or Jerry.

•	Training in Roboflow has allowed us to crop images and also change the contrast of certain images for better recognition by the model.

•	 Neural networks and machine learning have been used for these tasks and have obtained good results.

•	 Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for cartoon image detection as well.

# Jetson Nano Compatibility
• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Proposed System

Study basics of machine learning and image recognition.

Start with implementation

 ➢ Front-end development
 ➢ Back-end development
Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether object is Tom or jerry.

Collect the images from internet to interpret the object and suggest whether the object is Tom or Jerry.

# Methodology

Tom and Jerry Detection Module

This Module is divided into two parts:

1] Cartoon Image detection

Ability to detect the object in any input image or frame.

2] Classification Detection

• Classification of the object based on whether it is Tom or Jerry.

• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

•There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

• YOLOv5 was used to train and test our model for two classes Tom and Jerry. We trained it for 70 epochs and achieved an accuracy of approximately 90%.

# Installation
# Initial Setup
 Remove unwanted Applications.

sudo apt-get remove --purge libreoffice*

sudo apt-get remove --purge thunderbird*

#Create Swap file

sudo fallocate -l 10.0G /swapfile1

sudo chmod 600 /swapfile1

sudo mkswap /swapfile1

sudo vim /etc/fstab

#Cuda Configuration

vim ~/.bashrc


#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

source ~/.bashrc

# Udpade a System

sudo apt-get update && sudo apt-get upgrade

################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

sudo apt install curl

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

sudo python3 get-pip.py

sudo apt-get install libopenblas-base libopenmpi-dev

source ~/.bashrc

sudo pip3 install pillow

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl

mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo python3 -c "import torch; print(torch.cuda.is_available())"

# Installation of torchvision.

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install

# Clone yolov5 Repositories and make it Compatible with Jetson Nano.

cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/

sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0

# We used Google Colab And Roboflow
train your model on colab and download the weights and past them into yolov5 folder link of project

# Refrences

1]Roboflow :- https://roboflow.com/

2]Google images




