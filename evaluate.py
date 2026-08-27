import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from train import test_dataset, device

# Evaluate performance metrics of the best saved model, such as f1 score, precision, etc.


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
# Create DataLoader
# -------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# -------------------------
# Make predictions on ALL
# test data
# -------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# -------------------------
# Calculate metrics
# -------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

confusion = confusion_matrix(
    all_labels,
    all_predictions
)


# -------------------------
# Print results
# -------------------------

print("Evaluation on ALL test images")
print("------------------------------------")

print("Total test images:", len(test_dataset))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(confusion)

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "No Pothole",
            "Pothole"
        ],
        zero_division=0
    )
)


# -------------------------
# Evaluation Metrics Graph
# -------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    metrics,
    values,
    color=[
        "steelblue",
        "orange",
        "green",
        "red"
    ]
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance - All Test Images")

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "evaluation_metrics.png",
    dpi=300
)

plt.close()


# -------------------------
# Confusion Matrix Graph
# -------------------------

plt.figure(figsize=(7, 6))

sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Pothole",
        "Pothole"
    ],
    yticklabels=[
        "No Pothole",
        "Pothole"
    ]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - All Test Images")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


print("\nSaved evaluation_metrics.png")
print("Saved confusion_matrix.png")


# -------------------------
# Save metrics to TXT file
# -------------------------

report = classification_report(
    all_labels,
    all_predictions,
    target_names=[
        "No Pothole",
        "Pothole"
    ],
    zero_division=0
)

with open("evaluation_results.txt", "w") as file:

    file.write("MODEL EVALUATION RESULTS\n")
    file.write("========================\n\n")

    file.write(
        f"Total test images: {len(test_dataset)}\n\n"
    )

    file.write(
        f"Accuracy:  {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall:    {recall:.4f}\n"
    )

    file.write(
        f"F1 Score:  {f1:.4f}\n\n"
    )

    file.write("CONFUSION MATRIX\n")
    file.write("================\n")

    file.write(str(confusion))

    file.write("\n\nCLASSIFICATION REPORT\n")
    file.write("=====================\n")

    file.write(report)


print("Saved evaluation_results.txt")
