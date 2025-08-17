import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights
import logging

logger = logging.getLogger(__name__)

def setup_model(num_classes: int, device: torch.device):
    """
    Initializes the EfficientNetV2-L model with pre-trained weights
    and modifies its classifier head for the specified number of classes.
    """
    logger.info("Setting up EfficientNetV2-L model...")
    weights = EfficientNet_V2_L_Weights.DEFAULT
    model = models.efficientnet_v2_l(weights=weights)


    # first we are getting classifier in_features and then we are creating a new layer nn.linear using it as a classifier and giving it two classes and giving it the features we got from classifier
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("Model backbone initially frozen.")

    return model