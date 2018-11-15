import keras
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from model.VGG16 import VGG16
import pickle
num_cores = 4
num_GPU = 0
num_CPU = 4
# config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
#         inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
#         device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

BS = 120


# Model support
def train_model():
  train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode="nearest")

  test_datagen = ImageDataGenerator(rescale = 1./255)
  # load train data 
  training_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = (64, 64),
    batch_size = BS,
    shuffle=True,
    seed = 7,
    class_mode = 'categorical')
  # load validation data
  validation_set = test_datagen.flow_from_directory(
    'dataset/validation',
    target_size = (64, 64),
    batch_size = BS,
    shuffle=True,
    seed = 7,
    class_mode = 'categorical')

  print(validation_set.class_indices)
  print(validation_set.classes)
  print(validation_set.num_classes)

  imgs,labels = next(training_set)
  print(labels)

  file_Name = 'label_match.pickle'
  fileObject = open(file_Name,'wb') 
  # this writes the object a to the file named 'testfile'
  pickle.dump(validation_set.class_indices,fileObject)   
  # here we close the fileObject
  fileObject.close()


  nClasses = validation_set.num_classes

  print('nClasses',nClasses)
  model = VGG16().create_model((64, 64,3),nClasses)
  model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  model.fit_generator(training_set,
    steps_per_epoch = (1200/BS),
    nb_epoch = 20,
    validation_data = validation_set,
    verbose=1)


  # save the model to disk
  print("[INFO] serializing network...")
  model.save('lenet.model')
  model.save_weights('weights.h5')
    
if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    train_model()
