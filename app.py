import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fruit & Veg Classifier", layout="centered")
st.title("🍎 Fruit & Vegetable Classifier 🥦")
st.markdown("Upload an image to classify it!")

model_path = 'Image_classify.keras'

@st.cache_resource
def load_classification_model():
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_classification_model()

if model is None:
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
    'tomato', 'turnip', 'watermelon'
]

img_height = 180
img_width = 180

uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
    help="Upload an image of a fruit or vegetable"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((img_width, img_height))
        img_arr = tf.keras.utils.img_to_array(image)
        img_bat = tf.expand_dims(img_arr, 0)
        
        with st.spinner('Classifying...'):
            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict)
        predicted_class = data_cat[np.argmax(score)]
        confidence = np.max(score) * 100
        
        st.success(f"✅ **Prediction:** {predicted_class.title()}")
        st.info(f"🎯 **Confidence:** {confidence:.2f}%")
        
        top_3_indices = np.argsort(score[0])[-3:][::-1]
        st.markdown("### Top 3 Predictions:")
        for i, idx in enumerate(top_3_indices):
            confidence_score = score[0][idx] * 100
            st.write(f"{i+1}. **{data_cat[idx].title()}** - {confidence_score:.2f}%")
            
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("👆 Please upload an image file to get started!")
    
st.markdown("---")
st.markdown("### Supported Categories:")
st.write(", ".join([cat.title() for cat in data_cat]))