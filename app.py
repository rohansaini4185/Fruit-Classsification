import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.preprocessing import image
import cv2


model = tf.keras.models.load_model('Fresh-Rotten.h5')
st.write("""
         # Fresh-Rotten Fruit Classification by DataFolkz
         """
         )
st.write("Model will Predict whether a Fruit is Fresh or Rotten")
st.image("DataFolkz.png", width=700)
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


from PIL import Image, ImageOps


def import_and_predict(images, model):
        size = (150,150)    
        img = ImageOps.fit(images, size, Image.ANTIALIAS)
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        img = np.vstack([img])
        
        prediction = model.predict(img)        
        return prediction
    
      
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    classes = import_and_predict(image, model)
    
    if classes[0][0]==1:
        Output='Fresh Apple'
    elif classes[0][1]==1:
        Output='Fresh Banana'
    elif classes[0][2]==1:
        Output='Fresh Orange'
    elif classes[0][3]==1:
        Output='Rotten Apple'
    elif classes[0][4]==1:
        Output='Rotten Banana'
    elif classes[0][5]==1:
        Output='Rotten Orange'
    st.write("Prediction : ",Output)
