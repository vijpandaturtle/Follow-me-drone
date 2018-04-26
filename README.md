# **Follow Me Drone Project**

The purpose of this project was to use deep learning to enable a drone to autonomously follow it's intended target. For this project, I implemented and trained a Fully Convolutional Neural Net (FCN) to semantically segment every image(video frame) captured by the drone's camera to extract the location of the target.

## What is an FCN ?

A Fully Convolutional Neural Net (FCN) is an improved version of the regular Convolutional Neural Net (CNN) which includes the use of **1x1 convolutions**, **skip connections** and **transpose convolutions**. It is in some ways similar to a Convolutional Autoencoder, given that they both have a convolutional and a deconvolution(tranpose convolution) pair forming a single network. This network basically uses a regular CNN to form feature maps of the image and then uses the deconvolutional net to reconstruct  the image, while gaining a superior understanding of the image. An alternate name for an FCN is a segmentation network.

The architecture for the FCN is given below :

[image1]: ./images/download.png

![alt text][image1]

Below I will explain the network I implemented in the project notebook using the Keras library (Tensorflow backend) and all of the techniques I used which makes all the difference in the results. These techniques include hyperparameter tuning, skip connections and convolution filters. Setting the weights and bias will not be discussed however, because all this is already handled by keras which is a high level wrapper library.

**Note:** 1x1 convolutions are basically convolutions with kernel_size (filter size) set to 1x1. This is an easy way to make the feature map stack deeper without having to compromise on the amount of data available in that layer. By using a 1x1 convolutional layer we can retain spatial data from the previous data. Click [here](http://iamaaditya.github.io/2016/03/one-by-one-convolution/) for more information on 1x1 convolutional layers.

## Model

The model consists of two parts ; the encoder and the decoder. The encoder is your regular convolutional net having seperable convolutional layers. A seperable convolutional layer means we can split the kernel operation into multiple steps. A great resource that I used  to learn different types of convolutions is given [here.](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
I have also included code snippets consisting of the encoder and decoder architecture. As for the custom functions and syntax for defining layers, the keras documentation is provided [here.](https://keras.io/)

1. **Encoder block**

``` python
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
```
Consists of a single seperable convolutional layer, followed by batch normalization. Batch normalization is the process of shifting all the inputs to get a zero-mean and unit variance in small-sized    continuous chunks of data called batches. This helps us to take some liberties with weight initialization. To keep the network simple, I decided to use only one convolutional layer for the encoder. Encoder is your regular CNN, where it extracts features at each level thus giving your feature map. I have explained in detail about the functions of the layers in the upcoming sections.

2. **Decoder Block**

This is the main part of the network, which gives the segmentation network it's scene understanding capabilities. It is also referred to as the transpose convolution or deconvolutional network.

```python
   bilinear_upsampled = bilinear_upsample(small_ip_layer)
```
The first layer of this network is the bilinear upsampling layer. Upsampling is a basically an interpolation method. Interpolation is an estimation of a value within two known values in a given data sequence. Bilinear interpolation is generally used for 2D grids and works best in this situation. So, the end result of our bilinear upsampling gives us an output of (2H x 2W x D/2) given out inputs dimensions were (H x W x D).

```python
   concat_layers = layers.concatenate([bilinear_upsampled, large_ip_layer])
```
Next, we move on to talk about skip connections. **Skip connections**, is an easy method to get two different i.e feature maps from two disconnected layers to interact and infer from the other. The layers participating should be the deeper small layer which contains a lot feature data and the shallow large layer which has the relatively less feature data. The easiest implementation of the concept explained above, is the concatenation operation. Tensorflow and hence Keras, has a built in function for this.

```python
    conv = separable_conv2d_batchnorm(concat_layers, filters)
    conv_output_layer = separable_conv2d_batchnorm(conv, filters)
```

Lastly, this is followed by two separable convolutional layers along with batch normalization. These layers have their own significance in the semantic segmentation process or in simpler words in extracting data from an image. The beginning layers, extract low level features such as straight lines and simple shapes. The next layer/s record more complex features for examples curvature and more complex shapes. The level of complexity that each filter layer recognizes increases as we go deeper and then the final layers recognize objects as a whole. A convolutional net, if used for classification purposes usually uses fully connected layer with a softmax activation, to output class probabilities. In this case we don't need the class probabilities, but a segmented image, so we don't need fully connected layers.

**Note**:  In the above encoder blocks, you may have noticed that the max-pooling layer is absent. This is because using max-pooling layer leads to loss of data, so the transpose convolutional network will have difficulty in reconstructing the image. Refer to the below links to learn more about good filter sizes and strides for the convolutional layers.
1. [Guide To understanding CNNs](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/)
2. [Initializing filters](https://www.quora.com/What-is-are-the-method-s-for-initiating-choosing-filters-in-Convolutional-Neural-Networks)
3. [PyImagesearch](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

## Final Network

Now, we know the structure of the encoder and decoder block. The next step is to combine these two into a single fully convolutional network. Below is my implementation network. It is relatively small and simple and i'll walk you through each step of the network.

```python
    fcn_model_1 = encoder_block(inputs, 32, strides=2)
    fcn_model_2 = encoder_block(fcn_model_1, 64, strides=2)
```
Encoding the input image data using two encoding layers. The result of these two layers (our convolutional neural net ) is a deeper stack of feature maps.

```python
    fcn_model_3 = conv2d_batchnorm(fcn_model_2, 128, kernel_size=1, strides=1)
```
Follow this up with a regular convolutional layer plus batch_normalization layer.

```python
    fcn_model_4 = decoder_block(fcn_model_3,fcn_model_1, 64)
    final_layer = decoder_block(fcn_model_4, inputs, 32)
```
Now for the transpose convolutional net, we use two deocder blocks. You must have noticed that i'm using the same number of encoder and decoder layers, as shown above. That is because the image needs to be in it's original dimensions at the end of the decoder layers.

```python
   out = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(final_layer)
```
Lastly use a simple convolutional layer with a softmax activation to give us the segmented objects in the scene.

This is by no means the best network architecture. I made some optimizations along the way such as adding another encoder and decoder block to make the network a little deeper. My finally optimized architecture looked like this :

```python
    fcn_model_1 = encoder_block(inputs, 32, strides=2)
    fcn_model_2 = encoder_block(fcn_model_1, 64, strides=2)
    fcn_model_2_new = encoder_block(fcn_model_2, 128, strides=2)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    fcn_model_3 = conv2d_batchnorm(fcn_model_2_new, 512, kernel_size=1, strides=1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Block
    fcn_model_3 = decoder_block(fcn_model_3, fcn_model_2, 128)
    fcn_model_4 = decoder_block(fcn_model_3,fcn_model_1, 64)
    final_layer = decoder_block(fcn_model_4, inputs, 32)
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    out = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(final_layer)
```

Remember, a deeper network isn't always a good option, especially if your data isn't very complex. If you unnecessarily increase layers without keeping your dataset in mind, it will increase the training time drastically which might do more harm than good, and may not give you a good result. So, make sure your network is best suited to your data.  

**Note :** This segmentation network uses the intersection over union or IoU method for segmenting objects in the scene. The reason being, that it is more effective than bounding boxes, especially in a real-world environment. Here's an image illustrating the IoU method. Also click [here](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) for more info.

[image6]: ./images/iou.png
![alt text][image6]

## Training the Network

Below I will discuss in detail my different experiments with hyperparameter tuning and their effects on the training and validation loss.
Keep in mind that my actual number of training runs is much more than the number I will discuss, but the ones given below showed a significant improvement with smaller parameters.

**Params :**

learning_rate = 0.18,
batch_size = 128,
num_epochs = 20,
steps_per_epoch = 200,
validation_steps = 50,
workers = 2

[image2]: ./images/1st.png
![First Training Run][image2]

The training and validation seemed to converge well after these settings. But not without a few caveats. The training time was too long and both the losses seemed to converge too fast. So, in the next training runs, I played around with the number of epochs, learning rate and steps per epoch.

**Params :**

learning_rate = 0.12,
batch_size = 128,
num_epochs = 10,
steps_per_epoch = 200,
validation_steps = 50,
workers = 2

[image3]: ./images/2nd.png
![Second Training Run][image3]

Not too good either. The validation loss was higher than the training loss. I really thought, I could do better so I decided to lower the learning rate further. Point to note, I got pretty much the same results with 10 epochs and learning_rate of 0.09 and 200 steps, the only difference being that the validation loss reached a plateau after some time.

**Params :**

learning_rate = 0.06,
batch_size = 128,
num_epochs = 10,
steps_per_epoch = 100,
validation_steps = 50,
workers = 2

[image4]: ./images/3rd.png
![Third Training Run][image4]

Now this time tweaking the parameters drastically improved the performance of the network. It reached a new low of 0.09 !! Both the validation and training loss converged to the same point. There was only one thing that bothered me. The graph of the validation loss was quite uneven indicating a hint of unpredictability in the performance. It didn't take me long to realize that I hadn't done anything to improve the validation params at all. So, in the next few training runs, I stuck to modifying those params only.

**Params :**

learning_rate = 0.06,
batch_size = 128,
num_epochs = 10,
steps_per_epoch = 100,
validation_steps = 100,
workers = 2

[image5]: ./images/4th.png
![Fourth Training Run][image5]

The above training runs were the worst case runs in my training process, but I'm using this as an example to illustrate some optimization techniques, that will ultimately to your required result.
1. First, choose your learning rate wisely. If your learning rate is way too high, your loss will converge fast and plateau. This is a classic example of your model finding and settling for a local minima instead of a global minima.
Your ideal learning rate choices should be somewhere between **0.001-0.01**.You can go higher if you want, that entirely depends on the robustness of your dataset and architecture.

2. Along with learning rate, the number of epochs is also an important hyperparameter to optimize. This is especially necessary
when you have a low learning rate, because this gives more model enough time to converge. Following is the general trend to choose number of epochs.
( **Low learning_rate; Larger no of epochs** ::
**High learning_rate; Smaller no of epochs** )

3. Steps to training and validation indicates the number of images that would be used by the model at a time. It's good to choose reasonable number of images, as choosing a large number might increase training time significantly.

4. Finally, batch size. This is sometimes a tricky parameter to choose since it also significantly affects the overall performance of the model. The norm is to choose batch size as 32, 64, 128, 256 and so on. But this number is generally limited by the RAM on your local machine or compute instance. You can start from a small batch size and increase it, if you feel the need to do so.

After many trials of the above optimizations using educated guessed to guide me, these are the final set of params that gave me my end result.

```python
learning_rate = 0.01
batch_size = 64
num_epochs = 100
steps_per_epoch = 200
validation_steps = 50
workers = 8
```

### Results

This was my final training and validation loss. This was the loss I obtained in the middle of my training runs with the above hyperparameters, so I decided to stop it and use it for inference. For convenience purposes I also switched to tensorboard for visualiztion.

[final]: ./images/final.png
![alt text][final]

I used the sample training dataset for this project, which gave me satisfactory results. You can download this dataset by running the download.sh file in the data directory in your terminal. My final grade score was 0.44.

A brief analysis of my final metrics, which can be found in jupyter notebook, I conclude that my model was able to segment the background very well indicated by a high score in all three types of evaluation data. The model is also able to detect the hero among a crowd, provided he is at a reasonable distance. The model however, dosen't do very well with recognizing the target from a distance.

### Future Enhancements

This dataset performs decently well, on the data containing people. This should work in other classification problems, such as classifying a dog, cat etc. But it may need a couple optimizations, starting with increasing the number of layers and the number of filters. Also, one of the most important things, in training a deep learning model is a good dataset. For that I would use and recommend well known multi-class datasets such as the PASCAL VOC dataset and the IMAGENET dataset. Not only do these datasets have a large collection of data, they also don't need to be cleaned before use. As an afterthought, using such a simulator is a good idea to train the model to recognize images from a distance. As you can see in my final notebook the network isn't so good at recognizing the target from afar. That is something I intend to improve, by collecting additional data for it.
