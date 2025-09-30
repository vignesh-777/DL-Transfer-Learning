# DL – Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## THEORY
Transfer Learning is a technique where a pre-trained model (trained on a large dataset such as ImageNet) is used as a starting point for a different but related task. It leverages learned features from the original task to improve learning efficiency and performance on the new task.

VGG19 is a convolutional neural network with 19 layers. It consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. In transfer learning, we typically freeze the convolutional layers and retrain the final fully connected layers to match our dataset.

### Neural Network Model
**VGG19 Architecture for Transfer Learning:**

Input Image (224x224x3)
│
[Convolution + ReLU] x multiple layers
│
Max Pooling layers
│
Flatten Layer
│
Fully Connected Layer (4096)
│
Fully Connected Layer (4096)
│
Final Fully Connected Layer → num_classes (retrained)
│
Softmax Activation → Class Probabilities

## DESIGN STEPS
### STEP 1: Load and Preprocess Dataset
- Unzip the dataset and organize into train and test folders.  
- Resize all images to 224x224.  
- Convert images to tensors and optionally normalize them.

### STEP 2: Load Pretrained Model
- Load VGG19 model from `torchvision.models`.  
- Replace the last fully connected layer to match `num_classes` of your dataset.

### STEP 3: Define Loss and Optimizer
- Use `CrossEntropyLoss` for multi-class classification.  
- Use `Adam` optimizer to train only the final layer.

### STEP 4: Train the Model
- Freeze feature extractor layers.  
- Train only the final classifier for a few epochs.  
- Track training loss and validation loss for each epoch.

### STEP 5: Evaluate the Model
- Predict on the test set.  
- Compute accuracy, confusion matrix, and classification report.

### STEP 6: Predict on New Samples
- Select a test image.  
- Pass through the model and display predicted and actual class.






## PROGRAM

### Name: vignesh R

### Register Number: 212223240177

```python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Step 1: Load Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

!unzip -qq ./chip_data.zip -d data
train_dataset = datasets.ImageFolder(root="./data/dataset/train", transform=transform)
test_dataset = datasets.ImageFolder(root="./data/dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

```python
# Step 2: Load Pretrained Model
model = models.vgg19(pretrained=True)
num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False


```
```python
# Step 3: Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

```
```python
# Step 4: Train Model
def train_model(model, train_loader, test_loader, num_epochs=10):
    train_losses, val_losses = [], []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))
        model.train()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

train_model(model, train_loader, test_loader, num_epochs=10)
```
```python
# Step 5: Evaluate Model
def test_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

test_model(model, test_loader)
```
```python
# Step 6: Predict on Single Image
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        predicted = torch.argmax(prob, dim=1).item()
    class_names = dataset.classes
    plt.imshow(transforms.ToPILImage()(image))
    plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted]}")
    plt.axis("off")
    plt.show()
    print(f"Actual: {class_names[label]}, Predicted: {class_names[predicted]}")

predict_image(model, image_index=55, dataset=test_dataset)
predict_image(model, image_index=25, dataset=test_dataset)
```

### OUTPUT

## Training Loss
<img width="550" height="238" alt="image" src="https://github.com/user-attachments/assets/9adbff72-2240-460b-b58e-7174eb446a2e" />


## Validation Loss Vs Iteration Plot

<img width="807" height="646" alt="image" src="https://github.com/user-attachments/assets/5624e409-c7eb-4af0-a46d-c21f99bf81bc" />

## Confusion Matrix

<img width="884" height="710" alt="image" src="https://github.com/user-attachments/assets/6cb47ea6-41c4-4c92-9d2d-8a9db2b4e458" />

## Classification Report
<img width="587" height="239" alt="image" src="https://github.com/user-attachments/assets/62465e25-29d9-4759-ad07-17f122d199ac" />



### New Sample Data Prediction

<img width="433" height="448" alt="image" src="https://github.com/user-attachments/assets/377d038a-44aa-416e-a85c-56c975c3513c" />
<img width="483" height="481" alt="image" src="https://github.com/user-attachments/assets/e418301e-3d3b-4d1c-9518-edf3b63c1979" />


## RESULT
The model successfully classifies images from the dataset using transfer learning with VGG19. 
By freezing the convolutional layers and retraining the classifier, 
it achieves good accuracy while reducing training time.
