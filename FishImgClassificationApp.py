import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model with highest training accuaracy (MobileNetV2)
model = tf.keras.models.load_model('D:\AIML\FishImgClassificationProject\Model\FishImgClass_MobileNetV2.model.h5')

st.title("Fish Image Classifier")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Fish Image')

if st.button('Predict Fish Category',type='primary'):
    
    # Preprocess the image for model prediction
    resized_image = image.resize((224, 224)) # Example size
    image_array = np.array(resized_image) / 255.0 # Normalize

    # Define Fish image Categories/Classes
    class_names = { 0:'animal fish', 1:'animal fish bass', 2:'Black_sea_sprat',
                    3:'Gilt_head_bream', 4:'Hourse_mackerel',
                    5:'Red_mullet', 6:'Red_sea_bream',
                    7:'Sea_bass', 8:'Shrimp',
                    9:'Striped_red_mullet', 10:'Sea_food trout'}

    # Make predictions with the model for the input image
    prediction = model.predict(np.expand_dims(image_array, axis=0))
       
    # Display the predicted Fish image category/Class 
    st.subheader("Model Prediction:")
    st.write(f"Predicted Fish Category: {class_names[np.argmax(prediction)]}")