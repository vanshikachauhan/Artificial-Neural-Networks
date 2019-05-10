############################################################## NEURAL NETWORK WITH DENSE LAYERS ####################################################################################

#1.Import the keras modules and libraries
import keras
from keras import layers
from keras import models
from keras.utils import np_utils
from keras.datasets import mnist
import keras.metrics
from keras.preprocessing import image

#2. Load pre-shuffled MNIST data into train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print ("train_images shape:", train_images.shape)
print ("test_images shape:", test_images.shape)
print ("train_labels shape:", train_labels.shape)
print ("test_labels shape:", test_labels.shape)

#3. Preprocess input data
train_images =train_images.reshape(train_images.shape[0], 28* 28)
test_images = test_images.reshape(test_images.shape[0], 28*28)
print ("train_images shape after reshaping:", train_images.shape)
print ("test_images shape after reshaping:", test_images.shape)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images/= 255

#4. Preprocess class label Convert1-D class arrays to 10-D class matrices
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)
print ("train_labels shape after reshaping:", train_labels.shape)
print ("test_labels shape after reshaping:", test_labels.shape)

#5. Define model architecture
model = models.Sequential()
model.add(layers.Dense(30, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))

#6. summary of the model
model.summary()

#7. Compile model by declaring the loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='sgd',
metrics=["categorical_accuracy"])

#8. Fit model on training data
model.fit(train_images,train_labels,batch_size=128,epochs=5,verbose=0)

#9. Evaluate model on test data
score = model.evaluate(test_images, test_labels , verbose=0)
print('Test accuracy: %.2f%%', (score[1]* 100))
