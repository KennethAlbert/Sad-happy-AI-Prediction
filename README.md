# Emotion Classification App

## 📖 Project Overview

This is a deep learning-based web application that classifies facial expressions as either **Happy** or **Sad**. The application uses a Convolutional Neural Network (CNN) trained on facial image data to automatically detect and classify emotions from uploaded images.

## 🎯 What the Model Aims to Achieve

The primary goal of this project is to demonstrate the practical application of computer vision and deep learning for emotion recognition. Specifically, the model aims to:

- **Automatically classify** facial expressions into two emotion categories: Happy 😊 and Sad 😢
- **Provide real-time predictions** with confidence scores for uploaded images
- **Serve as an educational tool** showcasing end-to-end ML deployment from training to web application
- **Handle various image formats** and preprocess them for consistent model input
- **Offer interpretable results** with confidence breakdowns and technical details

### Use Cases
- Educational demonstrations of AI/ML capabilities
- Emotion analysis in user-generated content
- Basic sentiment detection from facial expressions
- Learning tool for computer vision and web deployment

## 🛠 Technologies & Libraries Used

### Core Machine Learning
- **TensorFlow 2.19.0** - Deep learning framework for model development and training
- **Keras** - High-level neural networks API (included with TensorFlow)
- **OpenCV (cv2)** - Computer vision library for image preprocessing
- **NumPy** - Numerical computing for array operations and data manipulation

### Web Application & Deployment
- **Streamlit 1.28.0** - Web framework for creating interactive ML web applications
- **Pillow (PIL)** - Image processing library for handling uploaded images

### Model Architecture
The CNN model consists of:
- **3 Convolutional Layers** with ReLU activation
- **MaxPooling Layers** for dimensionality reduction
- **Flatten Layer** to convert 2D features to 1D
- **Dense Layer** with 256 units and ReLU activation
- **Output Layer** with sigmoid activation for binary classification

### Training Specifications
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Input Shape**: 256x256x3 (RGB images)
- **Epochs**: 20
- **Train/Val/Test Split**: 70%/20%/10%

## 📁 Project Structure

```
emotion-classifier/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/               # Trained model files
│   ├── simple_emotion_model.h5
│   ├── simple_emotion_weights.weights.h5
│   └── emotion_weights.weights.h5
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place model files in the `models/` directory
4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Model Training (Optional)
The model was originally trained in Google Colab using:
- Custom dataset of happy and sad facial images
- Data augmentation and preprocessing
- GPU acceleration for faster training

## 💡 How to Use

1. **Launch the app** using `streamlit run app.py`
2. **Upload an image** using the file uploader (supports JPG, JPEG, PNG, BMP)
3. **View the prediction** - The app will display whether the person appears Happy or Sad
4. **Analyze confidence scores** - See the breakdown of prediction confidence
5. **Check technical details** - Expand the technical section for raw model outputs

## 🎨 Features

- **User-friendly Interface** - Clean, intuitive web interface
- **Real-time Predictions** - Instant emotion classification
- **Confidence Visualization** - Progress bars showing prediction certainty
- **Multiple File Support** - Handles various image formats
- **Technical Transparency** - Shows raw model outputs and processing details
- **Error Handling** - Comprehensive error messages and guidance

## 🔧 Technical Details

### Image Preprocessing Pipeline
1. **Format Conversion** - Handles RGB/RGBA/BGR conversions
2. **Resizing** - Standardizes images to 256x256 pixels
3. **Normalization** - Scales pixel values to [0, 1] range
4. **Batch Dimension** - Adds batch dimension for model input

### Model Performance
- **Binary Classification** between Happy and Sad emotions
- **Sigmoid Activation** for probability output
- **Threshold-based Decision** at 0.5 confidence level
- **Real-time Inference** optimized for web deployment

## 🌟 Future Enhancements

Potential improvements for the project:
- Support for more emotion categories (angry, surprised, neutral)
- Real-time webcam emotion detection
- Model retraining interface within the app
- Batch processing of multiple images
- Export functionality for results
- Mobile-responsive design improvements

## 📄 License

This project is intended for educational purposes. Please ensure proper attribution when using or modifying the code.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit pull requests.

---

**Note**: This application is designed for educational demonstrations and may not be suitable for clinical or high-stakes emotional analysis applications.
