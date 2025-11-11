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
    """
    Create the model architecture from scratch
    This matches your original Colab model exactly
    """
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
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

@st.cache_resource
def load_model_with_weights():
    """
    Create model architecture and load weights separately
    This avoids the InputLayer compatibility issue
    """
    st.sidebar.header("🔄 Loading Model")
    
    # Try loading strategies in order
    strategies = [
        # Strategy 1: Try loading the full simple model
        {
            'name': 'Full Simple Model',
            'file': 'models/simple_emotion_model.h5',
            'loader': lambda path: tf.keras.models.load_model(path, safe_mode=False)
        },
        # Strategy 2: Try loading weights into created architecture
        {
            'name': 'Weights + Architecture',
            'file': 'models/simple_emotion_weights.weights.h5',
            'loader': lambda path: load_weights_into_architecture(path)
        },
        # Strategy 3: Try original weights
        {
            'name': 'Original Weights',
            'file': 'models/emotion_weights.weights.h5',
            'loader': lambda path: load_weights_into_architecture(path)
        }
    ]
    
    for strategy in strategies:
        if os.path.exists(strategy['file']):
            try:
                st.sidebar.info(f"Trying: {strategy['name']}")
                model = strategy['loader'](strategy['file'])
                st.sidebar.success(f"✅ {strategy['name']} loaded!")
                return model
            except Exception as e:
                st.sidebar.error(f"❌ {strategy['name']} failed: {str(e)}")
                continue
    
    # Final fallback: create untrained model
    st.sidebar.warning("⚠️ No model files found - using untrained model")
    return create_model()

def load_weights_into_architecture(weights_path):
    """Load weights into newly created architecture"""
    model = create_model()
    model.load_weights(weights_path)
    return model

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert to numpy array
    img = np.array(image)
    
    # Convert to RGB if needed
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to 256x256
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize to [0, 1] and add batch dimension
    processed_img = np.expand_dims(img_resized / 255.0, 0)
    
    return processed_img, img

def display_file_status():
    """Display which model files are available"""
    with st.sidebar:
        st.header("📁 File Status")
        
        if not os.path.exists('models'):
            st.error("❌ 'models' folder not found!")
            return
        
        files = os.listdir('models')
        if files:
            st.success(f"✅ Found {len(files)} file(s):")
            for file in sorted(files):
                file_path = os.path.join('models', file)
                file_size = os.path.getsize(file_path) / 1024
                st.write(f"📄 {file} ({file_size:.1f} KB)")
        else:
            st.error("❌ No files in models folder")

def main():
    """Main function to run the Streamlit app"""
    
    # Display file status
    display_file_status()
    
    # Load model
    model = load_model_with_weights()
    
    if model:
        st.success("✅ Model is ready for predictions!")
        
        # File upload section
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear face image for emotion classification"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess and predict
                with st.spinner('🔍 Analyzing emotion...'):
                    processed_img, original_img = preprocess_image(image)
                    prediction = model.predict(processed_img, verbose=0)
                
                # Display results
                confidence = float(prediction[0][0])
                
                st.header("🎯 Prediction Results")
                
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
                    
                    st.write("**Confidence Breakdown:**")
                    st.progress(int(happy_conf))
                    st.write(f"😊 Happy: {happy_conf:.1f}%")
                    
                    st.progress(int(sad_conf))
                    st.write(f"😢 Sad: {sad_conf:.1f}%")
                
                # Show raw prediction value
                with st.expander("🔧 Technical Details"):
                    st.write(f"Raw prediction value: {prediction[0][0]:.6f}")
                    st.write(f"Decision threshold: 0.5")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        
        # Add some usage tips
        with st.expander("💡 Tips for best results"):
            st.markdown("""
            - **Use clear, well-lit face images**
            - **Front-facing photos work best**  
            - **Avoid heavy filters or editing**
            - **Ensure the face is clearly visible**
            - **Recommended size: 256x256 pixels or larger**
            """)
    
    else:
        st.error("❌ Could not initialize model")
        
        st.markdown("""
        ### 🚨 Setup Instructions
        
        1. **Run the fixed export code in Colab**
        2. **Download `compatible_models_fixed.zip`**
        3. **Extract the zip file**
        4. **Place these 3 files in the `models` folder:**
           - `simple_emotion_model.h5`
           - `simple_emotion_weights.weights.h5`
           - `emotion_weights.weights.h5`
        5. **Restart the app**
        """)

if __name__ == "__main__":
    main()