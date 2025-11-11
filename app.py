import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

st.title("🎭 Emotion Classification App")

def create_model():
    """Create model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try different model files
        model_files = [
            'models/simple_emotion_model.h5',
            'models/emotion_classifier.h5',
            'models/simple_emotion_weights.weights.h5'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                if 'weights' in model_file:
                    model = create_model()
                    model.load_weights(model_file)
                    return model
                else:
                    return tf.keras.models.load_model(model_file)
        
        # If no model files found, create empty model
        return create_model()
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return create_model()

def preprocess_image(image):
    """Preprocess image for prediction"""
    img = np.array(image)
    
    # Handle different image formats
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to 256x256
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize and add batch dimension
    processed_img = np.expand_dims(img_resized / 255.0, 0)
    
    return processed_img

def main():
    """Main app function"""
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Info")
        st.write("Upload a face image to classify emotion")
        
        # Check model files
        if os.path.exists('models'):
            files = os.listdir('models')
            if files:
                st.success(f"Found {len(files)} model file(s)")
            else:
                st.warning("No model files in models folder")
    
    # Load model
    model = load_model()
    
    if model:
        st.success("✅ Model loaded successfully!")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            try:
                # Display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Predict
                with st.spinner('Analyzing emotion...'):
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img, verbose=0)
                
                # Display results
                confidence = float(prediction[0][0])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if confidence > 0.5:
                        st.error(f"**😢 SAD**")
                        st.metric("Confidence", f"{confidence:.2%}")
                    else:
                        st.success(f"**😊 HAPPY**")
                        st.metric("Confidence", f"{(1-confidence):.2%}")
                
                with col2:
                    happy_conf = (1 - confidence) * 100
                    sad_conf = confidence * 100
                    
                    st.write("Confidence:")
                    st.progress(int(happy_conf))
                    st.write(f"Happy: {happy_conf:.1f}%")
                    
                    st.progress(int(sad_conf))
                    st.write(f"Sad: {sad_conf:.1f}%")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()