###################################Importing the libraries ##########################
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import load_model
           from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
##############Setting base_model by calling ResNet50 pre-trained model  ###########
HEIGHT = 150
WIDTH = 150
BATCH_SIZE = 8

base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))
############ path to the data ###############################################
TRAIN_DIR = "data1/train"
TEST_DIR = "data1/test"  
###############Data generator is used to get our data from our folders###########
train_datagen =  ImageDataGenerator( preprocessing_function=preprocess_input )
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE)

test_datagen=  ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=(HEIGHT,    
                           WIDTH), batch_size=BATCH_SIZE,class_mode=None,shuffle=False)
           
#################### Building actual model ##############################
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model

class_list = ["beggar1", "public1"]
FC_LAYERS = [1024, 1024]
dropout = 0.5
finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

NUM_EPOCHS = 3
BATCH_SIZE = 8
num_train_images = 17913

adam = Adam(lr=0.00001)
##################### compile the model ##############################
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])



######################### fitting the model ##############################
history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
                                       shuffle=True)

finetune_model.save("model_2.h5")
print("######################## Saved model to disk##########################")
#steps = 11654/8=1456
############################# evaluating the model ##########################
           probabilities =finetune_model.predict_generator(test_generator, 1456)
predicted_class_indices=np.argmax(probabilities,axis=1)
y_true = np.array([0] * 4675 + [1] * 6973)
y_pred = probabilities > 0.5
confusion_matrix(y_true, y_pred.argmax(axis=1))
