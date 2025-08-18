import os
import torchvision
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import EfficientNet_V2_L_Weights
import torch

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the exact calls order during thraing

class_names = ['female', 'male']

num_classes = len(class_names)

model_path = r"A:\\model_checkpoints\\gender_classifier_efficientnet_v2_l\\best_modeln.pth"


# load the pretrained efficient net v2-l
weights = EfficientNet_V2_L_Weights.DEFAULT
model = models.efficientnet_v2_l(weights=weights)

# Modify classifier for 5 classes (as in your training script)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)
model.to(device)



if not os.path.exists(model_path):
    print(f"❌ Error: model not found at: {model_path}")
    print("please ensure the 'model_part' variable point to your save '.pth' file. ")
    exit()

try: 
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"✅ Model weights laoded succesfully from: {model_path}")
except RuntimeError as e:
    print(f"❌ Error loading model state_dict: {e}")
    print("This might happen if the model architecture doesn't exactly match the saved state_dict.")
    print("Please ensure `model` definition (EfficientNetV2-L with 5-class classifier) matches your training script.")
    exit()

# Set model to evaluation mode (important for BatchNorm, Dropout layers)
model.eval()


# --3. define the same transform as validaton ---
# it's crucial to use the exact same preprocessing for inference as was used for validation
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Your val_transform resizes to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

print("✅ Image tranform difined (matching validation transform).")

# --- 4. Ask for image path ----
img_path = input("\n🖼️ Enter full path of the image to predict: ").strip()


if not os.path.exists(img_path):
    print(f"❌ Error: Image not found at: {img_path}")
    exit()

# --- 5. load and preprocess the image ---
try:
    image = Image.open(img_path).convert("RGB")
    print(f"✅ Image loaded: {img_path}")
except Exception as e:
    print(f"❌ Error laoding or opening image: {e}")
    exit()

# apply transform ad add batch dimension (unsqeuuze(0))
image = transform(image).unsqueeze(0).to(device)

# --- 6. predict --
with torch.no_grad(): # Disable gradient calculations for inference
    output_logits = model(image)
    # For multi-class classification with CrossEntropyLoss, output is logits.
    # To get probabilities, apply softmax.
    probabilities = torch.softmax(output_logits, dim=1)
    
    # Get the predicted class index and the maximum probability
    predicted_prob, predicted_idx = torch.max(probabilities, 1)

# Map the predicted index back to the class name
predicted_class = class_names[predicted_idx.item()]
confidence = predicted_prob.item()

print(f"\n--- Prediction Results ---")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
print(f"Raw Probabilities: {probabilities.cpu().squeeze().tolist()}") # Optional: show all probabilities



