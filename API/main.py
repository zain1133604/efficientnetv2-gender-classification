import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
# from torchvision.models import EfficientNet_V2_L_Weights # No longer needed for default weights
from contextlib import asynccontextmanager

# Initialize FastAPI app
app = FastAPI(
    title="Gender Classification API",
    description="A FastAPI application that classifies gender from an input image using EfficientNetV2-L."
)

# --- Global Variables for Model and Transformations ---
model = None
preprocess_transform = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['female', 'male'] # Based on your provided script

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown events for the FastAPI application.
    Model loading and cleanup occur here.
    """
    global model, preprocess_transform, device, class_names

    print(f"--- Application Startup ---")
    print(f"Attempting to load model on device: {device}")

    # Define the number of classes based on your class_names
    num_classes = len(class_names)

    # --- Model Definition ---
    # Load the EfficientNetV2-L model WITHOUT pre-trained ImageNet weights.
    # We are loading our own weights immediately after, so default weights are not needed
    model = models.efficientnet_v2_l(weights=None) # CHANGE THIS LINE

    # Modify classifier for the correct number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model.to(device)

    # --- Model Weights Loading (from your script) ---
    model_path = "./best_modeln.pth" # Ensure your .pth file is in the same directory!

    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights file not found at: {model_path}")
        print("Please ensure 'best_modeln.pth' is in the same directory as main.py.")
        raise RuntimeError("Model weights file not found.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"✅ Model weights loaded successfully from: {model_path}")
    except RuntimeError as e:
        print(f"❌ Error loading model state_dict: {e}")
        print("This might happen if the model architecture doesn't exactly match the saved state_dict.")
        raise RuntimeError(f"Failed to load model weights: {e}")

    # Set model to evaluation mode (important for BatchNorm, Dropout layers)
    model.eval()
    print("✅ Model set to evaluation mode.")

    # --- Define Image Transformations (from your script) ---
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    print("✅ Image transform defined (matching validation transform).")
    print(f"--- Application Ready ---")

    yield # The application will now start serving requests

    print(f"--- Application Shutdown ---")
    print("FastAPI application shutting down.")

# Initialize FastAPI app, passing the lifespan context manager
app = FastAPI(
    title="Gender Classification API",
    description="A FastAPI application that classifies gender from an input image using EfficientNetV2-L.",
    lifespan=lifespan # Pass the lifespan function here
)

# --- API Endpoint for Image Classification ---
@app.post("/classify_gender/")
async def classify_gender(file: UploadFile = File(...)):
    """
    Receives an image file, classifies gender using the loaded model,
    and returns the prediction.
    """
    if model is None or preprocess_transform is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Server may be starting up or encountered an error.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"✅ Image loaded from uploaded file: {file.filename}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image file: {e}")

    try:
        input_tensor = preprocess_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_logits = model(input_tensor)
            probabilities = torch.softmax(output_logits, dim=1)
            predicted_prob, predicted_idx = torch.max(probabilities, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence = predicted_prob.item()

        print(f"Prediction for {file.filename}: {predicted_class} with {confidence*100:.2f}% confidence")

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_gender": predicted_class,
            "confidence": f"{confidence*100:.2f}%",
            "raw_probabilities": probabilities.cpu().squeeze().tolist()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

# --- Simple Root Endpoint ---
@app.get("/")
async def root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Gender Classification API is running! Visit /docs for API documentation."}
