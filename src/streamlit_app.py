import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import glob
from model import load_model
from config import IMG_SIZE

st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    # Check both .keras and .h5 files
    model_files = glob.glob("models/*.keras") + glob.glob("models/*.h5") + glob.glob("*.keras") + glob.glob("*.h5")
    
    if not model_files:
        st.error("❌ No trained model found!")
        st.info("""To train a model, run the following commands in your terminal:
        
        ```bash
        cd src
        python main.py
        ```
        
        This will:
        1. Load and preprocess the training data
        2. Train an EfficientNetB3 model with transfer learning
        3. Save the trained model to the models/ directory
        
        Training may take 1-3 hours depending on your hardware.
        """)
        return None, None
    
    try:
        latest_model = max(model_files, key=os.path.getctime)
        st.success(f"✅ Loading model: {os.path.basename(latest_model)}")
        
        # Check if labels.csv exists
        if not os.path.exists("data/labels.csv"):
            st.error("❌ labels.csv not found in data/ directory!")
            st.info("Please ensure you have the Kaggle Dog Breed dataset in the data/ directory")
            return None, None
        
        model = load_model(latest_model)
        labels_csv = pd.read_csv("data/labels.csv")
        unique_breeds = np.unique(labels_csv["breed"].to_numpy())
        
        st.success(f"✅ Model loaded successfully with {len(unique_breeds)} breed classes")
        return model, unique_breeds
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess image to match training preprocessing"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize to 0-1 range
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_breed(model, image, unique_breeds):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_breeds = [unique_breeds[i] for i in top_5_indices]
    top_5_confidences = [predictions[0][i] for i in top_5_indices]
    
    return unique_breeds[predicted_class], confidence, top_5_breeds, top_5_confidences

def main():
    st.title("🐕 Dog Breed Classifier")
    st.markdown("Upload an image of a dog to identify its breed using our trained deep learning model!")
    
    model, unique_breeds = load_trained_model()
    
    if model is None:
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dog image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a dog for best results"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Classify Breed", type="primary"):
                with st.spinner("Analyzing image..."):
                    predicted_breed, confidence, top_5_breeds, top_5_confidences = predict_breed(
                        model, image, unique_breeds
                    )
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    st.success(f"**Predicted Breed:** {predicted_breed.replace('_', ' ').title()}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    st.subheader("Top 5 Predictions")
                    for i, (breed, conf) in enumerate(zip(top_5_breeds, top_5_confidences)):
                        st.write(f"{i+1}. **{breed.replace('_', ' ').title()}** - {conf:.2%}")
                    
                    st.subheader("Confidence Chart")
                    chart_data = pd.DataFrame({
                        'Breed': [breed.replace('_', ' ').title() for breed in top_5_breeds],
                        'Confidence': top_5_confidences
                    })
                    st.bar_chart(chart_data.set_index('Breed'))
    
    with col2:
        if uploaded_file is None:
            st.subheader("Model Information")
            st.info(f"Model loaded successfully!")
            st.info(f"Number of dog breeds: {len(unique_breeds)}")
            st.info("Supported formats: JPG, JPEG, PNG")
            
            st.subheader("Available Breeds")
            breed_list = [breed.replace('_', ' ').title() for breed in unique_breeds]
            breed_list.sort()
            
            search_breed = st.text_input("Search for a breed:")
            if search_breed:
                filtered_breeds = [breed for breed in breed_list if search_breed.lower() in breed.lower()]
                st.write(f"Found {len(filtered_breeds)} breeds matching '{search_breed}':")
                for breed in filtered_breeds[:10]:
                    st.write(f"• {breed}")
            else:
                st.write("Sample breeds:")
                for breed in breed_list[:10]:
                    st.write(f"• {breed}")
                st.write(f"... and {len(breed_list) - 10} more breeds")

if __name__ == "__main__":
    main()
