##################################################################CONVOLUTIONAL NEURAL NETWORK#########################################################################

#declare headers
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras.metrics
from keras.preprocessing import image
from quiver_engine import server

#loading mnist data sets
(train_image,train_lebels),(test_image,test_lebels)=mnist.load_data()

#loading values to labels and trainset
train_image=train_image.reshape(60000,28,28,1)
test_image=test_image.reshape(10000,28,28,1)

# convert to categorical
train_lebels= to_categorical(train_lebels)
test_lebels= to_categorical(test_lebels)

# float and normalization
train_image=train_image.astype('float32')/255
test_image=test_image.astype('float32')/255

#declare model architecture
network=models.Sequential()

#define the model layers
network.add(layers.Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
network.add(layers.Flatten())

network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(128,activation='relu'))
network.add(layers.Dense(10,activation='softmax'))

#compile the program
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


#fiting the data
network.fit(train_image,train_lebels,epochs=5,batch_size=128,verbose=1)

##evaluating the test set
##Score=network.evaluate(test_image,test_lebels,verbose=1)
##print("test accuracy:",Score[1]*100)

#external image testing
img1=image.load_img(path="one_black.png",grayscale=True,target_size=(28,28,1))

img1=image.img_to_array(img1)
tests1=img1.reshape((1,28,28,1))#in conv networks the should pe reshaped into 4d tensors
img_class=network.predict_classes(tests1)
prediction=img_class[0]
classname=img_class[0]
print("classname:",classname)

#displaying the digit and classifying digit
img1=img1.reshape((28,28))
plt.imshow(img1)
plt.title(classname)
plt.show()

#visualization by quiver engine
server.launch(network,input_folder="C:\\Users\\VANSHIKA CHAUHAN\\Desktop\\neural network",temp_folder="C:\\Users\\VANSHIKA CHAUHAN")

