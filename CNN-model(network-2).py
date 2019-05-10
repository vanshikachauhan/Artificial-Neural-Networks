################################################### NEURAL NETWORK WITH CONVOLUTION LAYERS AND DENSE LAYERS ###########################################################

#1.Import the keras modules and libraries
import keras
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from quiver_engine import server
from keras.utils import np_utils
from keras.datasets import mnist
import keras.metrics
from keras.preprocessing import image
import matplotlib.cm as cm


#2. Load pre-shuffled MNIST data into train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print ("train_images shape:", train_images.shape)
print ("test_images shape:", test_images.shape)
print ("train_labels shape:", train_labels.shape)
print ("test_labels shape:", test_labels.shape)


#3. Preprocess input data
train_images =train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
print ("train_images shape:", train_images.shape)
print ("test_images shape:", test_images.shape)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images/= 255


#4. Preprocess class labels convert 1-D class arrays to 10-D class matrices
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)
print ("train_labels shape:", train_labels.shape)
print ("test_labels shape:", test_labels.shape)


#5. Define model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
#6. Add a fully connected layer and then the output layer model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


#7. Model summary.
model.summary()


#8. Compile model by declaring the loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
metrics=["categorical_accuracy"])


#9. Fit model on training data
model.fit(train_images, train_labels,  batch_size=64, epochs=5, verbose=1)


#10. Evaluate model on test data
score = model.evaluate(test_images, test_labels , verbose=1)
print("Test score: %.2f%%", (score[0]* 100))
print('Test accuracy: %.2f%%', (score[1]* 100))


#11. load the input image forprediction
img=image.load_img(path="one_black.png",grayscale=True,target_size=(28,28,1))
img=image.img_to_array(img)
test_img=img.reshape((1,28,28,1))


#12. predict the classification
#classes = model.predict_classes(test_img, batch_size=32)
#classname=classes[0]
#print("class",classes)
#print("classname",classname)


#13. prepare image for ploting 
#img=img.reshape((28,28))
#plt.imshow(img,cmap=cm.Greys_r)
#plt.title(classname)


#14. plots the image
#plt.show()

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


#15. visualizing the model using quiver
#server.launch(model,input_folder="E:\Vanshika",temp_folder="E:\Vanshika\logs")
