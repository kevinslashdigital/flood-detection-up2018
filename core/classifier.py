# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2

def get_label(labels,predict):
  for key,value in labels.items():
    if value == predict:
        return key

def classify(image_path=None,input_shape=None):
  model = load_model('lenet.model')
  model.load_weights('weights.h5')
  if image_path != None:
    image = cv2.imread(image_path)
  # pre-process the image for classification
  image = cv2.resize(image,input_shape )
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  print("[INFO] classifying image...")

  file_Name = 'label_match.pickle'
  # load label match
  fileObject = open(file_Name,'rb')  
  # load the object from the file into var b
  label_match = pickle.load(fileObject)  

  # prediction = model.predict(image)
  # prediction = model.predict_classes(image)
  prediction = model.predict_proba(image)
  print('prediction',prediction)
  index = prediction.argmax(axis=-1)
  print('prediction',get_label(label_match,index),prediction[0][index])
  return prediction

if __name__ == "__main__":
  # construct the argument parse and parse the arguments
  classify('dataset/test/food/1_19.jpg',(64, 64))
