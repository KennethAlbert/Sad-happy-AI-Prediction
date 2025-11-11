import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import os

# Set page configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

st.title("🎭 Emotion Classification App")

@st.cache_resource
def load_model():
    """Load ONNX model"""
    try:
        model_path = 'models/emotion_model.onnx'
        if os.path.exists(model_path):
            session = ort.InferenceSession(model_path)
            st.sidebar.success("✅ ONNX model loaded!")
            return session
        else:
            st.sidebar.error("❌ Model file not found")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to numpy array
    img = np.array(image)
    
    # Convert to RGB if needed
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize to 256x256
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize to [0, 1] and add batch dimension
    processed_img = np.expand_dims(img_resized / 255.0, 0).astype(np.float32)
    
    return processed_img

def main():
    """Main function"""
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("Classifies facial expressions as Happy or Sad")
        
        # Check model file
        if os.path.exists('models/emotion_model.onnx'):
            st.success("✅ Model file found")
        else:
            st.error("❌ Model file missing")
    
    # Load model
    model = load_model()
    
    if model:
        st.success("🚀 Ready for predictions!")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, JPEG, PNG, BMP)",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            try:
                # Display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess and predict
                with st.spinner('🔍 Analyzing emotion...'):
                    processed_img = preprocess_image(image)
                    
                    # ONNX prediction
                    input_name = model.get_inputs()[0].name
                    prediction = model.run(None, {input_name: processed_img})[0]
                
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
                    
                    st.write("**Confidence Levels:**")
                    st.progress(int(happy_conf))
                    st.write(f"😊 Happy: {happy_conf:.1f}%")
                    
                    st.progress(int(sad_conf))
                    st.write(f"😢 Sad: {sad_conf:.1f}%")
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.write(f"Raw output: {prediction[0][0]:.6f}")
                    st.write(f"Threshold: 0.5")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        
        # Add usage tips
        with st.expander("💡 Tips for best results"):
            st.markdown("""
            - **Clear, well-lit face images work best**
            - **Front-facing photos are more accurate**  
            - **Avoid heavy filters or editing**
            - **Ensure the face is clearly visible**
            """)
    
    else:
        st.error("""
        ❌ **Model not loaded**
        
        Please ensure:
        1. Convert your model to ONNX using the Colab code
        2. Download `emotion_model.onnx`
        3. Place it in the `models` folder
        4. Redeploy the app
        """)

if __name__ == "__main__":
    main()