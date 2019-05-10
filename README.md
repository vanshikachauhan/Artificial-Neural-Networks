# Artificial-Neural-Networks
This project presents the image classification of hand-written digits from 0 to 9 and also the comparative analysis of traditional neural networks and convolutional neural networks (CNNs). The system is trained using MNIST dataset which consist of 60,000 training images and 10,000 test images of size (28 x 28 pixels) for each class of 0-9 digit. The system should be trained in such a way that it is able to recognise the hand-written gray-scale input digits.

I) ANN IMPLEMENTATION FOR HANDWRITTEN DIGIT CLASSIFICATION-(NETWORK-1)

PREPARING THE IMAGE DATA AND LABELS FOR NETWORK- Before training, the data was pre-processed by reshaping it into the shape the network expects and scaled it so that all values are in the [0, 1] interval. After that the labels were vectorized so that the output can be mapped and classified with required class of output. This is done by categorically encode the labels.
Reshapeing block diagram of training and testing images is following-


THE ARCHITECTURE OF NETWORK- The network consists of a sequence of three dense layers, which are densely connected, also called fully connected neural layers.
•	In the first layer various arguments which were passed were the number of neurons in the layer as 30, activation function as relu and the input shape as a column vector (28*28,1).

•	The second layer contained number of neurons as 20 and activation as relu but no input shape was passed as this layer takes the output of the previous layer.

•	The third (and last) layer was a dense layer with sigmoid activation function.Block diagram of network -1 architecture is following.

![20190510_211434](https://user-images.githubusercontent.com/43670329/57540888-65e96400-736b-11e9-8fff-b27c1425c367.jpg)
 
 
OUTPUT OF NETWORK-1

![Screenshot_20190510-230019_Docs](https://user-images.githubusercontent.com/43670329/57545728-e367a100-7378-11e9-8fa4-e84b4f895301.jpg)

II) CONVOLUTION NEURAL NETWORK-2 (CONVOLUTION NETWORK)

PREPARING THE IMAGE DATA AND LABELS FOR NETWORK-

The shape of the training images and testing images was changed, they were reshaped reshape into 4D array of (60000, 28, 28, 1) and (10000, 28, 28, 1) respectively for training images and testing images. And then normalized these images into [0, 1] of data type float32.Reshaping block diagram of 3d mnist data into 4d numpy array is following.

![20190510_211505](https://user-images.githubusercontent.com/43670329/57541756-608d1900-736d-11e9-9fc6-53ee3fe98e9a.jpg)



THE ARCHITECTURE OF NETWORK-2- The networks model of convolution network has three convolution layers, two maxpooling layers, one flatten layer and two dense layers.

1.	The first layer was the convolutional 2D layer with 32 filters of weight, filter size of (32, 32), activation relu and input shape of (28,28,1).

2.	Next layer was a maxpooling layer with stride size (2, 2). It extracts the maximum features from each stride of (2, 2).

3.	Third layer was the convolutional 2D layer with 64 filters of weight, filter size of (32, 32), activation is relu.

4.	Again next layer was a maxpooling layer with stride size (2, 2). It extracts the maximum features from each stride of (2, 2).

5.	Now the flatten layer was used as the next layers are dense layers. 

6.	Then dense layer with 64 neurons was added with activation relu.

7.	Last layer was a 10-way softmax layer, which means it had return an array of 10 probability scores (summing to 1). Each score was the probability that the current digit image belongs to one of the10 digit classes.Block diagram of network -2 architecture is following-

![20190510_211517 (1)](https://user-images.githubusercontent.com/43670329/57541744-5ec35580-736d-11e9-9257-d630cc65f10e.jpg)

                                        





PREDICT THE CLASSIFICATION-
A MS paint hand written digit has been generated of pixel size (28, 28) for the input test image.MS paint hand written digit is following-

                                                                       
In this network convolution layers are used so we have reshaped this image as 4D vector (1,28,28,1).

The classification was predicted for the input test image by passing test image as an argument in predict_classes function. And the class name was printed.The class predicts the input test image as 1.




PLOT THE INPUT TEST IMAGE AND CLASSNAME-

The test image was reshaped back to its original pixel size i.e. (28,28). And then image and class name as title was shown by show function of matplotlib.
The code for plotting is in appendix-2 and the output is following.Input image graph with title as classification

![20190510_230105](https://user-images.githubusercontent.com/43670329/57545729-e367a100-7378-11e9-9511-3d6bd0d6ffc0.jpg)

The title shows that the predicted class is 1. 
 
OUTPUT OF NETWORK-2-
![20190510_230042](https://user-images.githubusercontent.com/43670329/57545731-e4003780-7378-11e9-957b-35869e95b3aa.jpg)
                  










 
              
