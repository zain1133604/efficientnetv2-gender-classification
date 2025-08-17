# import os
# import torchvision
# from torchvision import models, transforms
# from PIL import Image
# import torch.nn as nn
# from torchvision.models import EfficientNet_V2_L_Weights
# import torch

# # set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define the exact calls order during thraing

# class_names = ['female', 'male']

# num_classes = len(class_names)

# model_path = r"A:\\model_checkpoints\\gender_classifier_efficientnet_v2_l\\best_modeln.pth"


# # load the pretrained efficient net v2-l
# weights = EfficientNet_V2_L_Weights.DEFAULT
# model = models.efficientnet_v2_l(weights=weights)

# # Modify classifier for 5 classes (as in your training script)
# num_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_features, num_classes)
# model.to(device)



# if not os.path.exists(model_path):
#     print(f"❌ Error: model not found at: {model_path}")
#     print("please ensure the 'model_part' variable point to your save '.pth' file. ")
#     exit()

# try: 
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     print(f"✅ Model weights laoded succesfully from: {model_path}")
# except RuntimeError as e:
#     print(f"❌ Error loading model state_dict: {e}")
#     print("This might happen if the model architecture doesn't exactly match the saved state_dict.")
#     print("Please ensure `model` definition (EfficientNetV2-L with 5-class classifier) matches your training script.")
#     exit()

# # Set model to evaluation mode (important for BatchNorm, Dropout layers)
# model.eval()


# # --3. define the same transform as validaton ---
# # it's crucial to use the exact same preprocessing for inference as was used for validation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)), # Your val_transform resizes to 224x224
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                         [0.229, 0.224, 0.225])
# ])

# print("✅ Image tranform difined (matching validation transform).")

# # --- 4. Ask for image path ----
# img_path = input("\n🖼️ Enter full path of the image to predict: ").strip()


# if not os.path.exists(img_path):
#     print(f"❌ Error: Image not found at: {img_path}")
#     exit()

# # --- 5. load and preprocess the image ---
# try:
#     image = Image.open(img_path).convert("RGB")
#     print(f"✅ Image loaded: {img_path}")
# except Exception as e:
#     print(f"❌ Error laoding or opening image: {e}")
#     exit()

# # apply transform ad add batch dimension (unsqeuuze(0))
# image = transform(image).unsqueeze(0).to(device)

# # --- 6. predict --
# with torch.no_grad(): # Disable gradient calculations for inference
#     output_logits = model(image)
#     # For multi-class classification with CrossEntropyLoss, output is logits.
#     # To get probabilities, apply softmax.
#     probabilities = torch.softmax(output_logits, dim=1)
    
#     # Get the predicted class index and the maximum probability
#     predicted_prob, predicted_idx = torch.max(probabilities, 1)

# # Map the predicted index back to the class name
# predicted_class = class_names[predicted_idx.item()]
# confidence = predicted_prob.item()

# print(f"\n--- Prediction Results ---")
# print(f"Predicted Class: {predicted_class}")
# print(f"Confidence: {confidence*100:.2f}%")
# print(f"Raw Probabilities: {probabilities.cpu().squeeze().tolist()}") # Optional: show all probabilities



# Save this code as a new Python file, e.g., 'quantize_model_flexible.py',
# You can place it anywhere, but typically in your project's root or a scripts folder.

import torch
import torch.nn as nn
from torchvision import models
from torch.quantization import quantize_dynamic
import os

# --- CONFIGURATION (EDIT THESE TWO LINES) ---
# 1. Full path to your ORIGINAL model file (including its name, e.g., 'model.pth')
#    Based on your previous input, this is where your original model is located.
ORIGINAL_MODEL_FILE_PATH = r"A:\\model_checkpoints\\gender_classifier_efficientnet_v2_l\\best_modeln.pth"

# 2. Directory where you want to SAVE the NEW quantized model file.
#    This should be your 'gender-classification' folder for easy pushing to Hugging Face.
OUTPUT_SAVE_DIRECTORY = r"A:\\fast_api\\gender-classification"
# ---------------------------------------------

# Define the number of output classes for your model
# Assuming 'female' and 'male' for gender classification
num_classes = 2

# Derived output filename for the quantized model (will be created in OUTPUT_SAVE_DIRECTORY)
output_quantized_model_name = "best_modeln_quantized.pth"
output_quantized_model_full_path = os.path.join(
    OUTPUT_SAVE_DIRECTORY,
    output_quantized_model_name
)

# --- 1. Instantiate the Original Model Architecture ---
# Use the same architecture as your original model (EfficientNetV2-L)
# We initialize it without pre-trained ImageNet weights, as we'll load our own.
print("1. Instantiating EfficientNetV2-L model architecture...")
model = models.efficientnet_v2_l(weights=None)

# Modify the classifier head to match your number of classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)
print(f"   Model classifier adapted for {num_classes} classes.")

# --- 2. Load Your Original Trained Weights ---
print(f"2. Loading original weights from: {ORIGINAL_MODEL_FILE_PATH}")
try:
    # Load the state_dict onto CPU for quantization (GPU not strictly needed for this step)
    model.load_state_dict(torch.load(ORIGINAL_MODEL_FILE_PATH, map_location='cpu', weights_only=True))
    print("   Original model weights loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Failed to load original model weights. Check the path and file integrity.")
    print(f"Error details: {e}")
    exit() # Exit if loading fails

# Set the model to evaluation mode (important for quantization)
model.eval()
print("   Model set to evaluation mode.")

# --- 3. Apply Dynamic Quantization ---
print("3. Applying dynamic quantization (converting weights to lower precision)...")
# Dynamic quantization targets 'Linear' layers (common in classifiers)
# and converts them to int8, reducing memory usage.
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("   Dynamic quantization applied.")

# --- 4. Save the Quantized Model ---
print(f"4. Saving quantized model to: {output_quantized_model_full_path}")
try:
    # Ensure the output directory exists
    os.makedirs(OUTPUT_SAVE_DIRECTORY, exist_ok=True)
    # Save only the state_dict of the quantized model
    torch.save(quantized_model.state_dict(), output_quantized_model_full_path)
    print("✅ Quantized model saved successfully!")
    print(f"   Original model size: {os.path.getsize(ORIGINAL_MODEL_FILE_PATH) / (1024 * 1024):.2f} MB")
    print(f"   Quantized model size: {os.path.getsize(output_quantized_model_full_path) / (1024 * 1024):.2f} MB")
except Exception as e:
    print(f"❌ ERROR: Failed to save quantized model.")
    print(f"Error details: {e}")

print("Quantization script finished.")