#!/usr/bin/env python3
"""
Dog Breed Classification - Streamlit Web Interface
This is the main Streamlit application for dog breed classification.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import glob
import sys
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
IMG_SIZE = 300  # Same as training pipeline
MODEL_EXTENSIONS = ['.h5', '.keras', '.pt']

@st.cache_resource
def load_model_and_classes():
    """
    Load the trained model and class indices.
    Exits gracefully if model is missing.
    """
    # Check for model files
    model_files = []
    for ext in MODEL_EXTENSIONS:
        model_files.extend(glob.glob(f"models/*{ext}"))
        model_files.extend(glob.glob(f"*{ext}"))
    
    if not model_files:
        st.error("❌ No trained model found!")
        st.markdown("""
        ### Model Not Found
        
        Please train a model first by running:
        
        ```bash
        python train_model.py
        ```
        
        This will:
        1. Load and preprocess the dog breed dataset
        2. Train a MobileNetV2 model with transfer learning
        3. Save the trained model to `models/model.h5`
        4. Save class indices to `models/class_indices.json`
        
        Training typically takes 30-60 minutes depending on your hardware.
        """)
        st.stop()
        
    # Find the latest model
    latest_model = max(model_files, key=os.path.getctime)
    st.success(f"✅ Found model: {os.path.basename(latest_model)}")
    
    # Load model
    try:
        if latest_model.endswith('.pt'):
            # PyTorch model loading would go here
            st.error("PyTorch models not supported yet. Please use TensorFlow/Keras models.")
            st.stop()
        else:
            # Load TensorFlow/Keras model
            model = tf.keras.models.load_model(latest_model)
            st.success(f"✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Load class indices
    class_indices_path = "models/class_indices.json"
    if os.path.exists(class_indices_path):
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            # Convert to list of class names in correct order
            class_names = [None] * len(class_indices)
            for breed, idx in class_indices.items():
                class_names[idx] = breed
            st.success(f"✅ Loaded {len(class_names)} breed classes")
        except Exception as e:
            st.error(f"❌ Error loading class indices: {str(e)}")
            st.stop()
    else:
        st.error("❌ class_indices.json not found in models/ directory!")
        st.markdown("""
        ### Class Indices Not Found
        
        The `class_indices.json` file is missing. This file should be generated during training.
        Please retrain the model using:
        
        ```bash
        python train_model.py
        ```
        """)
        st.stop()
    
    return model, class_names

def preprocess_image(image):
    """
    Preprocess image identically to training pipeline.
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to match training size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize to [0, 1] range (same as training: rescale=1./255)
    image_array = image_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_breed(model, image, class_names):
    """
    Run inference on the preprocessed image.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_breed = class_names[predicted_class_idx]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_breeds = [class_names[i] for i in top_5_indices]
    top_5_confidences = [predictions[0][i] for i in top_5_indices]
    
    return predicted_breed, confidence, top_5_breeds, top_5_confidences

def format_breed_name(breed_name):
    """
    Format breed name for display (replace underscores with spaces and title case).
    """
    return breed_name.replace('_', ' ').title()

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🐕 Dog Breed Classifier")
    st.markdown("""
    Upload an image of a dog to identify its breed using our trained deep learning model!
    
    This model uses transfer learning with MobileNetV2 and can classify over 100 different dog breeds.
    """)
    
    # Load model and class names
    model, class_names = load_model_and_classes()
    
    # Sidebar with model information
    with st.sidebar:
        st.header("Model Information")
        st.info(f"**Classes:** {len(class_names)}")
        st.info(f"**Image Size:** {IMG_SIZE}x{IMG_SIZE}")
        st.info(f"**Supported Formats:** JPG, JPEG, PNG")
        
        # Show available breeds
        st.subheader("Available Breeds")
        breed_list = [format_breed_name(breed) for breed in class_names]
        breed_list.sort()
        
        # Search functionality
        search_term = st.text_input("Search breeds:", "")
        if search_term:
            filtered_breeds = [breed for breed in breed_list if search_term.lower() in breed.lower()]
            st.write(f"Found {len(filtered_breeds)} matches:")
            for breed in filtered_breeds[:15]:  # Show max 15 results
                st.write(f"• {breed}")
        else:
            st.write("Sample breeds:")
            for breed in breed_list[:15]:
                st.write(f"• {breed}")
            if len(breed_list) > 15:
                st.write(f"... and {len(breed_list) - 15} more")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a dog image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a dog for best results. Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔍 Classify Breed", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Make prediction
                        predicted_breed, confidence, top_5_breeds, top_5_confidences = predict_breed(
                            model, image, class_names
                        )
                        
                        # Store results in session state for display in second column
                        st.session_state.prediction_results = {
                            'predicted_breed': predicted_breed,
                            'confidence': confidence,
                            'top_5_breeds': top_5_breeds,
                            'top_5_confidences': top_5_confidences
                        }
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    with col2:
        st.subheader("Prediction Results")
        
        # Display results if available
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            # Main prediction
            st.success(f"**Predicted Breed:** {format_breed_name(results['predicted_breed'])}")
            st.info(f"**Confidence:** {results['confidence']:.2%}")
            
            # Top 5 predictions
            st.subheader("Top 5 Predictions")
            for i, (breed, conf) in enumerate(zip(results['top_5_breeds'], results['top_5_confidences'])):
                st.write(f"{i+1}. **{format_breed_name(breed)}** - {conf:.2%}")
            
            # Bar chart of top 5 probabilities
            st.subheader("Confidence Distribution")
            chart_data = pd.DataFrame({
                'Breed': [format_breed_name(breed) for breed in results['top_5_breeds']],
                'Confidence': results['top_5_confidences']
            })
            
            # Create bar chart
            st.bar_chart(chart_data.set_index('Breed'))
            
            # Additional visualization with matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(results['top_5_breeds'])), results['top_5_confidences'])
            ax.set_yticks(range(len(results['top_5_breeds'])))
            ax.set_yticklabels([format_breed_name(breed) for breed in results['top_5_breeds']])
            ax.set_xlabel('Confidence')
            ax.set_title('Top 5 Breed Predictions')
            
            # Add value labels on bars
            for i, (bar, conf) in enumerate(zip(bars, results['top_5_confidences'])):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{conf:.2%}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.info("Upload an image and click 'Classify Breed' to see results here.")
            
            # Show example usage
            st.markdown("""
            ### How to use:
            1. Upload a clear image of a dog
            2. Click the "Classify Breed" button
            3. View the prediction results and confidence scores
            
            ### Tips for best results:
            - Use clear, well-lit images
            - Ensure the dog is the main subject
            - Avoid heavily cropped or blurry images
            - Images with single dogs work better than multiple dogs
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🐕 Dog Breed Classifier | Built with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
