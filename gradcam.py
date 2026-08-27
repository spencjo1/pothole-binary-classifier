import numpy as np
import cv2
import random
import torch
import torch.nn as nn

from train import test_dataset, device

from torchvision.models import resnet18, ResNet18_Weights

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -------------------------
# Load the trained model
# -------------------------

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

model.fc = nn.Linear(
    in_features=512,
    out_features=2
)

model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)

model = model.to(device)
model.eval()


# -------------------------
# Select one random image
# -------------------------

random_index = random.randint(
    0,
    len(test_dataset) - 1
)

image, label = test_dataset[random_index]

input_tensor = image.unsqueeze(0).to(device)


# -------------------------
# Make prediction
# -------------------------

with torch.no_grad():

    outputs = model(input_tensor)

    _, predicted = torch.max(
        outputs,
        1
    )

predicted_label = predicted.item()


# -------------------------
# Print information
# -------------------------

class_names = [
    "No Pothole",
    "Pothole"
]

print("--------------------------------")
print("Random Test Image")
print("--------------------------------")

print("Test image index:", random_index)

print(
    "Actual label:",
    label,
    f"({class_names[label]})"
)

print(
    "Predicted label:",
    predicted_label,
    f"({class_names[predicted_label]})"
)


# -------------------------
# Set up Grad-CAM
# -------------------------

target_layers = [
    model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# -------------------------
# Generate Grad-CAM
# -------------------------

targets = [
    ClassifierOutputTarget(predicted_label)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)

grayscale_cam = grayscale_cam[0]


# -------------------------
# Convert image back to RGB
# -------------------------

rgb_image = (
    image
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)

mean = np.array([
    0.485,
    0.456,
    0.406
])

std = np.array([
    0.229,
    0.224,
    0.225
])

rgb_image = (
    std * rgb_image
    + mean
)

rgb_image = np.clip(
    rgb_image,
    0,
    1
)


# -------------------------
# Save original image
# -------------------------

original_image = (
    rgb_image * 255
).clip(
    0,
    255
).astype(
    np.uint8
)

cv2.imwrite(
    "original_image.jpg",
    cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )
)


# -------------------------
# Save Grad-CAM
# -------------------------

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)

cv2.imwrite(
    "gradcam_result.jpg",
    cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )
)


print("\nSaved original_image.jpg")
print("Saved gradcam_result.jpg")
