# Artificial-Neural-Networks
This project presents the image classification of hand-written digits from 0 to 9 and also the comparative analysis of traditional neural networks and convolutional neural networks (CNNs). The system is trained using MNIST dataset which consist of 60,000 training images and 10,000 test images of size (28 x 28 pixels) for each class of 0-9 digit. The system should be trained in such a way that it is able to recognise the hand-written gray-scale input digits.

I) ANN IMPLEMENTATION FOR HANDWRITTEN DIGIT CLASSIFICATION

PREPARING THE IMAGE DATA AND LABELS FOR NETWORK- Before training, the data was pre-processed by reshaping it into the shape the network expects and scaled it so that all values are in the [0, 1] interval. After that the labels were vectorized so that the output can be mapped and classified with required class of output. This is done by categorically encode the labels.
Reshapeing block diagram of training and testing images is following-


THE ARCHITECTURE OF NETWORK- The network consists of a sequence of three dense layers, which are densely connected, also called fully connected neural layers.
•	In the first layer various arguments which were passed were the number of neurons in the layer as 30, activation function as relu and the input shape as a column vector (28*28,1).

•	The second layer contained number of neurons as 20 and activation as relu but no input shape was passed as this layer takes the output of the previous layer.

•	The third (and last) layer was a dense layer with sigmoid activation function.Block diagram of network -1 architecture is following.

![20190510_211434](https://user-images.githubusercontent.com/43670329/57540888-65e96400-736b-11e9-8fff-b27c1425c367.jpg)
 
SUMMARY OF THE NETWORK
![20190510_211449 (1)](https://user-images.githubusercontent.com/43670329/57541755-608d1900-736d-11e9-8bf5-2dcb72bfdecb.jpg)




NETWORK COMPILATION AND TRAINING- 
 network is compiled by using loss function as categorical_crossentropy, optimizer as sgd and metrics as categorical_accuracy and is trained by using with number of epochs as 5, batch size128.

TRAINING AND TESTING ACCURACY-

The total time for training data was 13 seconds, the loss was 1.0209. And the training accuracy was 66.81% as shown below.


 

The test accuracy comes out to be 73.49%.

 

II) CONVOLUTION NEURAL NETWORK-2 (CONVOLUTION NETWORK)

PREPARING THE IMAGE DATA AND LABELS FOR NETWORK-

The shape of the training images and testing images was changed, they were reshaped reshape into 4D array of (60000, 28, 28, 1) and (10000, 28, 28, 1) respectively for training images and testing images. And then normalized these images into [0, 1] of data type float32.Reshaping block diagram of 3d mnist data into 4d numpy array is following.
![20190510_211505](https://user-images.githubusercontent.com/43670329/57541756-608d1900-736d-11e9-9fc6-53ee3fe98e9a.jpg)


![20190510_211505](https://user-images.githubusercontent.com/43670329/57541756-608d1900-736d-11e9-9fc6-53ee3fe98e9a.jpg)

THE ARCHITECTURE OF NETWORK- The networks model of convolution network has three convolution layers, two maxpooling layers, one flatten layer and two dense layers.

1.	The first layer was the convolutional 2D layer with 32 filters of weight, filter size of (32, 32), activation relu and input shape of (28,28,1).

2.	Next layer was a maxpooling layer with stride size (2, 2). It extracts the maximum features from each stride of (2, 2).

3.	Third layer was the convolutional 2D layer with 64 filters of weight, filter size of (32, 32), activation is relu.

4.	Again next layer was a maxpooling layer with stride size (2, 2). It extracts the maximum features from each stride of (2, 2).

5.	Now the flatten layer was used as the next layers are dense layers. 

6.	Then dense layer with 64 neurons was added with activation relu.

7.	Last layer was a 10-way softmax layer, which means it had return an array of 10 probability scores (summing to 1). Each score was the probability that the current digit image belongs to one of the10 digit classes.Block diagram of network -2 architecture is following-

![20190510_211517 (1)](https://user-images.githubusercontent.com/43670329/57541744-5ec35580-736d-11e9-9257-d630cc65f10e.jpg)

                                        

SUMMARY OF THE NETWORK

![20190510_211528](https://user-images.githubusercontent.com/43670329/57541750-5f5bec00-736d-11e9-894b-c262f366fddf.jpg)




                                             
NETWORK COMPILATION AND TRAINING- 
We compiled our network by using loss function as categorical_crossentropy, optimizer as rmsprop and metrics as categorical_accuracy and is trained by using with number of epochs as 5, batch size128.

TRAINING AND TESTING ACCURACY-

The total time for training data was 13 seconds, the loss was 1.0209. And the training accuracy was 96.70% as shown below.


 

The test accuracy comes out to be 97.63%.

 

PREDICT THE CLASSIFICATION-
A MS paint hand written digit has been generated of pixel size (28, 28) for the input test image.MS paint hand written digit is following-

                                                                       
In this network convolution layers are used so we have reshaped this image as 4D vector (1,28,28,1).

The classification was predicted for the input test image by passing test image as an argument in predict_classes function. And the class name was printed.The class predicts the input test image as 2.


PLOT THE INPUT TEST IMAGE AND CLASSNAME-

The test image was reshaped back to its original pixel size i.e. (28,28). And then image and class name as title was shown by show function of matplotlib.
The code for plotting is in appendix-2 and the output is following.Input image graph with title as classification

![20190510_211557 (2)](https://user-images.githubusercontent.com/43670329/57541746-5f5bec00-736d-11e9-87ee-bcbbb3454cd5.jpg)

          

The title shows that the predicted class is 2. 
                  
VISUALIZING THE VARIOUS LAYERS OF NETWORK-2-

Al the activation maps were visualized by launching server of quiver engine.And the output is following:

![20190510_211618](https://user-images.githubusercontent.com/43670329/57541745-5f5bec00-736d-11e9-9c4e-a98ed3b84d22.jpg)




![20190510_211618](https://user-images.githubusercontent.com/43670329/57541745-5f5bec00-736d-11e9-9c4e-a98ed3b84d22.jpg)
![20190510_211557 (2)](https://user-images.githubusercontent.com/43670329/57541746-5f5bec00-736d-11e9-87ee-bcbbb3454cd5.jpg)
![20190510_211528](https://user-images.githubusercontent.com/43670329/57541750-5f5bec00-736d-11e9-894b-c262f366fddf.jpg)
![20190510_211741](https://user-images.githubusercontent.com/43670329/57541751-5ff48280-736d-11e9-8bf0-6b34b92f41c8.jpg)
![20190510_211449 (1)](https://user-images.githubusercontent.com/43670329/57541755-608d1900-736d-11e9-8bf5-2dcb72bfdecb.jpg)
![20190510_211505](https://user-images.githubusercontent.com/43670329/57541756-608d1900-736d-11e9-9fc6-53ee3fe98e9a.jpg)
![20190510_211557](https://user-images.githubusercontent.com/43670329/57541757-6125af80-736d-11e9-9c98-6d2833e2378a.jpg)






 
                            Fig58-Visualization of input test image by quiver
