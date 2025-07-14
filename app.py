import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
import numpy as np
import os

# Configuration - MODIFIED FOR HUGGING FACE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Use a relative path or load from Hugging Face Hub
MODEL_PATH = 'efficientnet_b0_medium_augmentation.pth'

# Model configuration for EfficientNet B0
MODEL_CONFIG = {
    'model_fn': models.efficientnet_b0,
    'feature_dim': 1280,
    'input_size': 224
}

def create_model(num_classes, pretrained=False):
    """Create EfficientNet B0 model with custom classifier"""
    model = MODEL_CONFIG['model_fn'](pretrained=pretrained)
    model.classifier[1] = nn.Linear(MODEL_CONFIG['feature_dim'], num_classes)
    return model

def get_transforms():
    """Get image preprocessing transforms"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform

def load_model():
    """Load the trained model"""
    try:
        num_classes = 2  # Adjust this based on your dataset
        model = create_model(num_classes, pretrained=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict(image):
    """Make prediction on uploaded image"""
    try:
        transform = get_transforms()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        probabilities = probabilities.cpu().numpy()[0]
        class_names = ['negatif', 'positif']
        
        results = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        return results
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}

# Load model at startup
print("🚀 Loading model...")
model = load_model()

if model is None:
    print("❌ Failed to load model. Please check the model path.")
else:
    print("✅ Model loaded successfully!")

# Create Gradio interface
def create_interface():
    custom_css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .gr-button {
        color: white;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border: none;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .gr-button:hover {
        background: linear-gradient(90deg, #45a049, #4CAF50);
    }
    """

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Label(num_top_classes=2, label="Predictions"),
        title="🤖 EfficientNet B0 Image Classifier",
        description="""
        **Transfer Learning Model Deployment**
        This EfficientNet B0 model classifies images into two categories.
        """,
        css=custom_css,
        examples=[  # Add example images if you have them
            # "example1.jpg",
            # "example2.jpg"
        ]
    )
    return iface

# For Hugging Face Spaces, we don't launch immediately
interface = create_interface()

# This is needed for Hugging Face Spaces
if __name__ == "__main__":
    interface.launch()