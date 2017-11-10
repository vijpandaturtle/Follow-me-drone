# **Follow Me Drone Project**

The purpose of this project was to use deep learning to enable a drone to autonomously follow it's intended target. For this project, I implemented and trained a Fully Convolutional Neural Net (FCN) to semantically segment every image(video frame) captured by the drone's camera to extract the location of the target.

## What is an FCN ?

A Fully Convolutional Neural Net (FCN) is an improved version of the regular Convolutional Neural Net (CNN) which includes the use of **1x1 convolutions**, **skip connections** and **transpose convolutions**. It is in some ways similar to a Convolutional Autoencoder, given that they both have a convolutional and a deconvolution(tranpose convolution) pair forming a single network. This network basically uses a regular CNN to form feature maps of the image and then uses the deconvolutional net to reconstruct  the image, while gaining a superior understanding of the image. An alternate name for an FCN is a segmentation network.

The architecture for the FCN is given below :

[image1] : ./images/download.png

![alt text][image1]

Below I will explain the network I implemented in the project notebook using the Keras library (Tensorflow backend) and all of the techniques I used which makes all the difference in the results. These techniques include hyperparameter tuning, skip connections and convolution filters. Setting the weights and bias will not be discussed however, because all this is already handled by keras which is a high level wrapper library.

## Model

The model consists of two parts ; the encoder and the decoder. The encoder is your regular convolutional net having seperable convolutional layers. A seperable convolutional layer means we can split the kernel operation into multiple steps. A great resource that I used  to learn different types of convolutions is given [here.](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
I have also included code snippets consisting of the encoder and decoder architecture. As for the custom functions and syntax for defining layers, the keras documentation is provided [here.](https://keras.io/)

1. **Encoder block**

``` python
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
```
Consists of a single seperable convolutional layer, followed by batch normalization. Batch normalization is the process of shifting all the inputs to get a zero-mean and unit variance in small-sized    continuous chunks of data called batches. This helps us to take some liberties with weight initialization. To keep the network simple, I decided to use only one convolutional layer for the encoder.

2. **Decoder Block**

This is the main part of the network, which gives the segmentation network it's scene understanding capabilities. It is also referred to as the transpose convolution or deconvolutional network.

```python
   bilinear_upsampled = bilinear_upsample(small_ip_layer)
```
The first layer of this network is the bilinear upsampling layer. Upsampling is a basically an interpolation method. Interpolation is an estimation of a value within two known values in a given data sequence. Bilinear interpolation is generally used for 2D grids and works best in this situation. So, the end result of our bilinear upsampling gives us an output of (2H x 2W x D/2) given out inputs dimensions were (H x W x D).

```python
   concat_layers = layers.concatenate([bilinear_upsampled, large_ip_layer])
```
Next, we move on to talk about skip connections. Skip connections, is an easy method to get two different i.e feature maps from two disconnected layers to interact and infer from the other. The layers participating should be the deeper small layer which contains a lot feature data and the shallow large layer which has the relatively less feature data. The easiest implementation of the concept explained above, is the concatenation operation. Tensorflow and hence Keras, has a built in function for this.

```python
    conv = separable_conv2d_batchnorm(concat_layers, filters)
    conv_output_layer = separable_conv2d_batchnorm(conv, filters)
```

Lastly, this is followed by two separable convolutional layers along with batch normalization.

**Note**:  In the above encoder blocks, you may have noticed that the max-pooling layer is absent. This is because using max-pooling layer leads to loss of data, so the transpose convolutional network will have difficulty in reconstructing the image. Refer to the below links to learn more about good filter sizes and strides for the convolutional layers.
1. [Guide To understanding CNNs](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/)
2. [Initializing filters](https://www.quora.com/What-is-are-the-method-s-for-initiating-choosing-filters-in-Convolutional-Neural-Networks)
3. [PyImagesearch](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)
