import os
import torch
import torch.nn as nn
from torchvision.models import densenet201
from torchvision import transforms as T
from PIL import Image

# 1. Define Configuration Constants
MODEL_WEIGHTS_PATH = os.path.join('models', 'densenet201_final.pth')
TEST_IMAGES_DIR = 'test_images'
DIAGNOSTIC_THRESHOLD = 0.5  # Slide this down to 0.35 or 0.40 based on your team's clinical goals

# 2. Rebuild the Exact Neural Network Architecture
def load_clinical_model():
    print("Initializing DenseNet-201 network architecture...")
    model = densenet201()
    
    # Modify classifier head to precisely match training parameters
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    
    # Map model to CPU explicitly for local machine execution compatibility
    print(f"Loading weights from {MODEL_WEIGHTS_PATH} onto CPU...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()  # Set layers to evaluation mode to lock batch normalization and dropout
    return model

# 3. Match the Exact Image Transformation Pipeline from Training
def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 4. Batch Processing Execution Flow
def run_batch_inference():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Weights file not found at {MODEL_WEIGHTS_PATH}. Did you download it from Colab?")
        return

    model = load_clinical_model()
    transform = get_transforms()
    
    # Supported image formats
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"No valid ultrasound images found in the '{TEST_IMAGES_DIR}/' folder.")
        return

    print(f"\nFound {len(images)} images to process. Running diagnostics...\n")
    print(f"{'Image Filename':<30} | {'Predicted Classification':<25} | {'Malignant Probability'}")
    print("-" * 80)

    with torch.no_grad():  # Disables gradient tracking to save RAM and speed up execution
        for img_name in images:
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            
            try:
                # Open and convert to RGB (handles grayscale or CMYK anomalies)
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]
                
                # Forward pass prediction
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                malignant_prob = probabilities[0, 1].item()
                
                # Determine binary diagnosis class assignment
                diagnosis = "MALIGNANT (High Risk)" if malignant_prob >= DIAGNOSTIC_THRESHOLD else "BENIGN (Low Risk)"
                
                print(f"{img_name:<30} | {diagnosis:<25} | {malignant_prob * 100:.2f}%")
            
            except Exception as e:
                print(f"{img_name:<30} | Error processing file: {str(e)}")

if __name__ == '__main__':
    run_batch_inference()