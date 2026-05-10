# Crop Disease Detection with Explainable AI 2025

This project implements a Convolutional Neural Network (CNN) based on MobileNetV2 to classify 38 plant diseases across 14 crops using the PlantVillage dataset. It achieves high accuracy and includes Explainable AI (XAI) using Grad-CAM to highlight the leaf regions responsible for predictions, supporting practical decision-making in agriculture.

## Features
- **Deep Learning Model:** Transfer learning using MobileNetV2 trained on the PlantVillage dataset.
- **Explainable AI:** Grad-CAM visualization generates heatmaps over the original leaf images.
- **Web App:** Interactive UI built with Streamlit to easily test images.

## Setup Instructions

### 1. Training the Model (Google Colab)
Since training deep learning models requires significant compute (GPU), we recommend using Google Colab.
1. Upload the `train_model_colab.ipynb` file to [Google Colab](https://colab.research.google.com/).
2. Run all cells in the notebook. It will automatically download the dataset, train the MobileNetV2 model, and evaluate it.
3. At the end of the notebook, two files will be generated and downloaded:
   - `plant_disease_model.h5`
   - `class_indices.json`
4. Place these two files in the root directory of this project locally.

### 2. Running the Local Streamlit App
Ensure you have Python installed locally.

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
6. Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).
7. Upload a crop leaf image to see the prediction and Grad-CAM heatmap!

## Project Structure
- `train_model_colab.ipynb`: Jupyter Notebook for training the model.
- `app.py`: Streamlit web application.
- `gradcam.py`: Logic for generating Grad-CAM heatmaps.
- `requirements.txt`: Python dependencies.
