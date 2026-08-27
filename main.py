# PROOF OF CONCEPT

import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from train import device
from locator import get_image_metadata

# An outline of how this can be used.


# -------------------------
# Settings
# -------------------------


#Test image WITH pothole
image_path = "all_data/aAAwuvUBxwBSvhR.JPG"

#Test image with NO pothole
#image_path = "all_data/OWpzucPasZzDRSd.JPG"

# Use simulated GPS data for testing
TEST_MODE = True


# -------------------------
# Load trained model
# -------------------------

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

model.fc = nn.Linear(
    in_features=512,
    out_features=2
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        weights_only=True
    )
)

model = model.to(device)
model.eval()


# -------------------------
# Load and transform image
# -------------------------

image = Image.open(image_path)

transform = weights.transforms()

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0).to(device)


# -------------------------
# Run AI prediction
# -------------------------

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted = torch.argmax(
        probabilities,
        dim=1
    )

predicted_label = predicted.item()

confidence = probabilities[
    0,
    predicted_label
].item()


# -------------------------
# Class names
# -------------------------

class_names = [
    "No Pothole",
    "Pothole"
]

prediction = class_names[predicted_label]


# -------------------------
# Get image metadata
# -------------------------

metadata = get_image_metadata(
    image_path
)


# -------------------------
# Use simulated GPS
# if real GPS is unavailable
# -------------------------

if TEST_MODE:

    if metadata["latitude"] is None:

        metadata["latitude"] = 42.4850

    if metadata["longitude"] is None:

        metadata["longitude"] = -83.0270


# -------------------------
# Print results
# -------------------------

print()
print("========================================")
print("       POTHOLE DETECTION SYSTEM")
print("========================================")

print()

print("Image:")
print(
    " ",
    metadata["filename"]
)

print()

print("AI Prediction:")
print(
    " ",
    prediction
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)

print()

print("Location:")

if (
    metadata["latitude"] is not None
    and metadata["longitude"] is not None
):

    print(
        " Latitude:",
        metadata["latitude"]
    )

    print(
        " Longitude:",
        metadata["longitude"]
    )

else:

    print(
        " GPS data unavailable"
    )

print()

print("Date Taken:")

if metadata["date_time"] is not None:

    print(
        " ",
        metadata["date_time"]
    )

else:

    print(
        " Date unavailable"
    )

print()
print("========================================")


# -------------------------
# Simulated city notification
# -------------------------

if predicted_label == 1:

    print()
    print("⚠ POTHOLE DETECTED")

    print(
        "Creating pothole report..."
    )

    # -------------------------
    # Create report
    # -------------------------

    with open(
        "pothole_report.txt",
        "w"
    ) as file:

        file.write(
            "POTHOLE REPORT\n"
        )

        file.write(
            "===============\n\n"
        )

        file.write(
            "Status: Pothole Detected\n\n"
        )

        file.write(
            f"Image: {metadata['filename']}\n"
        )

        file.write(
            f"Model Confidence: "
            f"{confidence * 100:.2f}%\n\n"
        )

        file.write(
            "LOCATION\n"
        )

        file.write(
            "--------\n"
        )

        if (
            metadata["latitude"] is not None
            and metadata["longitude"] is not None
        ):

            file.write(
                f"Latitude: "
                f"{metadata['latitude']}\n"
            )

            file.write(
                f"Longitude: "
                f"{metadata['longitude']}\n"
            )

        else:

            file.write(
                "GPS data unavailable\n"
            )

        file.write("\n")

        file.write(
            "DATE TAKEN\n"
        )

        file.write(
            "----------\n"
        )

        if metadata["date_time"] is not None:

            file.write(
                f"{metadata['date_time']}\n"
            )

        else:

            file.write(
                "Date unavailable\n"
            )

        file.write("\n")

        file.write(
            "NOTIFICATION\n"
        )

        file.write(
            "------------\n"
        )

        file.write(
            "This report represents a "
            "proof-of-concept notification "
            "that could be sent to the "
            "appropriate city department.\n"
        )

    print(
        "Saved pothole_report.txt"
    )

    print(
        "A real implementation could "
        "send this report to the city."
    )

else:

    print()
    print("✓ No pothole detected.")
