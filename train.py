import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd

train_set = pd.read_csv("train_ids_labels.csv")

train_has_pothole = train_set[train_set['Label'] == 1]
train_no_pothole = train_set[train_set['Label'] == 0]


weights = ResNet18_Weights.DEFAULT

# Some images are flipped to better generalize and train the model
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    weights.transforms()
])

validation_transform = weights.transforms()
model = resnet18(weights=weights)

#Split the data into an 80% 10% 10% split
train_data, temp_data = train_test_split(train_set, test_size=.2, stratify=train_set["Label"], random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=.5, stratify=temp_data["Label"], random_state=42)

# Preprocesses all of the pothole images and labels

class PotholeDataset(Dataset):

    def __init__(self, dataframe, image_directory, transform):
        self.dataframe = dataframe
        self.image_directory = image_directory
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_id = row['Image_ID']
        label = row['Label']

        image_path = self.image_directory + "/" + image_id + ".JPG"

        image = Image.open(image_path)
        image = self.transform(image)

        return image, label

# Split into train, hold, and test
train_dataset = PotholeDataset(train_data, 'all_data', train_transform)
validation_dataset = PotholeDataset(validation_data, 'all_data', validation_transform)
test_dataset = PotholeDataset(test_data, 'all_data', validation_transform)

#print(len(train_dataset))

#image, label = train_dataset[0]
#print(image.shape)
#print(label)


train_loader = DataLoader(batch_size=64, dataset=train_dataset, shuffle=True)
validation_loader = DataLoader(batch_size=64, dataset=validation_dataset, shuffle=False)
test_loader = DataLoader(batch_size=64, dataset=test_dataset, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)





#---------------------------THE CODE BELOW BELOW IS EXCLUSIVELY FOR TRAINING THE MODEL-------------------------------------------------------
#-------------------SINCE THE BEST PERFORMANCE OF THE MODEL IS SAVED, THIS IS COMMENTED OUT.-------------------------------------------------




# model.fc = nn.Linear(in_features=512, out_features=2)
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 5
# best_accuracy = 0.0
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print("Training Loss: ", running_loss / len(train_loader))


#     model.eval()
#     correct = 0
#     total = 0
#     running_validation_loss = 0.0

#     with torch.no_grad():
#         for images, labels in validation_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             validation_loss = criterion(outputs, labels)
#             running_validation_loss += validation_loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print("Validation Loss: ", running_validation_loss / len(validation_loader))

#     validation_accuracy = correct / total
#     print("Validation accuracy: ", validation_accuracy)
#     if validation_accuracy > best_accuracy:
#         best_accuracy = validation_accuracy
#         torch.save(model.state_dict(), 'best_model.pth')
#         print("New best model!")

# model.load_state_dict(torch.load("best_model.pth", weights_only=True))


# test_ids = pd.read_csv("test_ids_only.csv")

# class FinalTestDataset(Dataset):

#     def __init__(self, dataframe, image_directory, transform):
#         self.dataframe = dataframe
#         self.image_directory = image_directory
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]

#         image_id = row['Image_ID']

#         image_path = self.image_directory + "/" + image_id + ".JPG"

#         image = Image.open(image_path)
#         image = self.transform(image)

#         return image, image_id


# final_test_dataset = FinalTestDataset(
#     test_ids,
#     'all_data',
#     validation_transform
# )

# final_test_loader = DataLoader(
#     batch_size=64,
#     dataset=final_test_dataset,
#     shuffle=False
# )

# model.fc = nn.Linear(in_features=512, out_features=2)

# model.load_state_dict(torch.load("best_model.pth", weights_only=True))

# model = model.to(device)

# model.eval()

# predictions = []

# with torch.no_grad():

#     for images, image_ids in final_test_loader:

#         images = images.to(device)

#         outputs = model(images)

#         _, predicted = torch.max(outputs, 1)

#         for image_id, prediction in zip(image_ids, predicted):
#             predictions.append([image_id, prediction.item()])

# print(predictions[:10])
# print("Number of predictions:", len(predictions))

# predictions_df = pd.DataFrame(predictions, columns=["Image_ID", "Label"])

# predictions_df.to_csv("test_predictions.csv", index=False)

# print(predictions_df.head())

