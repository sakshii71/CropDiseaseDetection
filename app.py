import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import io
import matplotlib.pyplot as plt
from gradcam import get_last_conv_layer_name, make_gradcam_heatmap, overlay_heatmap
from disease_info import DISEASE_INFO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

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

def create_pdf_report(original_img, heatmap_img, predicted_class, confidence, info_dict):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    Story = []

    # Title
    Story.append(Paragraph("Crop Disease Detection System", styles['Title']))
    Story.append(Spacer(1, 12))

    # Prediction
    Story.append(Paragraph(f"<b>Prediction:</b> {predicted_class}", styles['Heading2']))
    Story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2%}", styles['Normal']))
    Story.append(Spacer(1, 12))

    # Disease Info
    if info_dict:
        Story.append(Paragraph("<b>Description:</b> " + info_dict.get('description', ''), styles['Normal']))
        Story.append(Spacer(1, 6))
        Story.append(Paragraph("<b>Causes:</b> " + info_dict.get('causes', ''), styles['Normal']))
        Story.append(Spacer(1, 6))
        Story.append(Paragraph("<b>Treatment/Prevention:</b> " + info_dict.get('treatment', ''), styles['Normal']))
        Story.append(Spacer(1, 12))

    # Save images to temp buffers to embed
    orig_io = io.BytesIO()
    original_img.save(orig_io, format='JPEG')
    orig_io.seek(0)
    
    heat_img_pil = Image.fromarray(heatmap_img)
    heat_io = io.BytesIO()
    heat_img_pil.save(heat_io, format='JPEG')
    heat_io.seek(0)

    # Add images
    Story.append(Paragraph("<b>Uploaded Image</b>", styles['Heading3']))
    Story.append(RLImage(orig_io, width=250, height=250))
    Story.append(Spacer(1, 12))
    
    Story.append(Paragraph("<b>Explainable AI (Grad-CAM)</b>", styles['Heading3']))
    Story.append(RLImage(heat_io, width=250, height=250))

    doc.build(Story)
    buffer.seek(0)
    return buffer

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
        
        # --- NEW FEATURE 1: Disease Info & Treatment ---
        info = DISEASE_INFO.get(predicted_class, None)
        st.subheader("Disease Information & Treatment")
        if info:
            with st.expander("Show Details", expanded=True):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Causes:** {info['causes']}")
                st.write(f"**Treatment/Prevention:** {info['treatment']}")
        else:
            st.info("No detailed information available for this class.")

        # --- NEW FEATURE 2: Top 3 Predictions Chart ---
        st.subheader("Top 3 Predictions")
        # Ensure we don't try to get top 3 if there are fewer than 3 classes
        top_k = min(3, len(class_names))
        top_k_indices = np.argsort(preds[0])[-top_k:][::-1]
        top_k_classes = [class_names[i] for i in top_k_indices]
        top_k_probs = [preds[0][i] for i in top_k_indices]

        fig, ax = plt.subplots(figsize=(6, 3))
        y_pos = np.arange(len(top_k_classes))
        ax.barh(y_pos, top_k_probs, align='center', color='skyblue')
        ax.set_yticks(y_pos, labels=top_k_classes)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Confidence')
        st.pyplot(fig)

        # Grad-CAM
        with st.spinner("Generating Explainable AI Heatmap..."):
            try:
                last_conv, base_model_name = get_last_conv_layer_name(model)
                heatmap = make_gradcam_heatmap(img_batch, model, last_conv, base_model_name, pred_index)
                
                img_array_original_size = np.array(img)
                superimposed_img = overlay_heatmap(img_array_original_size, heatmap, alpha=0.5)
                
                with col2:
                    st.image(superimposed_img, caption='Grad-CAM Heatmap', use_container_width=True)
                    st.info("The red/yellow regions indicate the areas the AI focused on most to make its prediction.")
                    
                # --- NEW FEATURE 3: PDF Report Generation ---
                pdf_buffer = create_pdf_report(img, superimposed_img, predicted_class, float(confidence), info)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name="crop_disease_report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Could not generate Grad-CAM heatmap: {e}")
