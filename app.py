import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from gradcam import get_last_conv_layer_name, make_gradcam_heatmap, overlay_heatmap

st.set_page_config(page_title="Crop Disease AI", layout="wide")

st.title("🌿 Crop Disease Detection with Explainable AI")
st.write("Upload a crop leaf image to detect diseases. The model will also provide a Grad-CAM heatmap to explain which parts of the leaf influenced its decision.")

@st.cache_resource
def load_model_and_classes():
    model_path = 'plant_disease_model.h5'
    class_indices_path = 'class_indices.json'
    
    if not os.path.exists(model_path) or not os.path.exists(class_indices_path):
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    with open(class_indices_path, 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_classes()

if model is None:
    st.warning("⚠️ Model or class indices not found. Please train the model using Colab notebook, and download `plant_disease_model.h5` and `class_indices.json` to this directory.")
else:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read the image
        img = Image.open(uploaded_file).convert('RGB')
        
        # Display layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption='Uploaded Image', use_container_width=True)
            st.write("Processing...")
            
        # Preprocess the image
        img_array = np.array(img)
        img_resized = tf.image.resize(img_array, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0) # Add batch dimension
        
        # Prediction
        preds = model.predict(img_batch)
        pred_index = np.argmax(preds[0])
        confidence = preds[0][pred_index]
        predicted_class = class_names[pred_index]
        
        st.subheader(f"Prediction: **{predicted_class}**")
        st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
        
        # Grad-CAM
        with st.spinner("Generating Explainable AI Heatmap..."):
            try:
                last_conv, base_model_name = get_last_conv_layer_name(model)
                heatmap = make_gradcam_heatmap(img_batch, model, last_conv, base_model_name, pred_index)
                
                # Resize original image to 224x224 for consistent visualization if it's too large/small
                # Or we can resize heatmap to original image size
                img_array_original_size = np.array(img)
                superimposed_img = overlay_heatmap(img_array_original_size, heatmap, alpha=0.5)
                
                with col2:
                    st.image(superimposed_img, caption='Grad-CAM Heatmap', use_container_width=True)
                    st.info("The red/yellow regions indicate the areas the AI focused on most to make its prediction.")
            except Exception as e:
                st.error(f"Could not generate Grad-CAM heatmap: {e}")
