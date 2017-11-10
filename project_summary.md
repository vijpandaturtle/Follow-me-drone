# **Follow Me Drone Project**

The purpose of this project was to use deep learning to enable a drone to autonomously follow it's intended target. For this project, I implemented and trained a Fully Convolutional Neural Net (FCN) to semantically segment every image(video frame) captured by the drone's camera to extract the location of the target.

## What is an FCN ?

A Fully Convolutional Neural Net (FCN) is an improved version of the regular Convolutional Neural Net (CNN) which includes the use of 1x1 convolutions, skip connections and transpose convolutions. It is in some ways similar to a Convolutional Autoencoder, given that they both have a convolutional and a deconvolution(tranpose convolution) pair forming a single network. This network basically uses a regular CNN to form feature maps of the image and then uses the deconvolutional net to reconstruct  the image, while gaining a superior understanding of the image. An alternate name for an FCN is a segmentation network.

The architecture for the FCN is given below :
![alt text][./images/download.png]

Below I will explain the network I implemented in the project notebook using the Keras library (Tensorflow backend) and all of the techniques I used which makes all the difference in the results. These techniques include hyperparameter tuning, skip connections and convolution filters. Setting the weights and bias will not be discussed however, because all this is a already handled by keras which is a high level wrapper library.

## Model
