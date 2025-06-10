# Trained with 1000 good and 1000 bad synthetic images

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm

# Setup
data_dir = r"c:\Users\halvo\Documents\Boat-Classification-Master\good_and_bad_synthetic"
save_path = "/cluster/home/halvorbb/project_test/efficientnetb3_best_good_bad_classifier.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Lower batch size if memory issues arise
batch_size = 8

epochs = 25
val_ratio = 0.2
early_stopping = 50
learning_rate = 3e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentations (experimental values)
transform = transforms.Compose([
    transforms.RandomResizedCrop(456, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.RandomRotation(15)
    ], p=0.3),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
    ], p=0.4),
    transforms.RandomApply([
        transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    ], p=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Windows requires __name__ == "__main__" loop because of multiprocessing issues
if __name__ == "__main__":
    # Dataset and dataloader logic
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load pre-trained ImageNet weights for EfficientNet-B3
    model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=2)

    # Dropout rate also experimental, usually between 0.2 and 0.5. Added here to final layer
    # https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, 2)
    )

    # Move model to cuda if available
    model = model.to(device)

    # Object function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # Learning rate scheduler to reduce learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Early stopping parameters and initialization
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # Training
    # https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    # Epoch loop
    for epoch in range(epochs):
        model.train()
        epoch_train_loss, epoch_train_corrects = 0.0, 0

        # Batch loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backpropagation and optimization
            loss.backward()
            optimizer.step()

            # Count loss per batch
            epoch_train_loss += loss.item() * inputs.size(0)

            # Count correct predictions per batch
            preds = torch.argmax(outputs, 1)
            epoch_train_corrects += (preds == labels).sum().item()

        # Average train loss and accuracy for the epoch
        train_loss = epoch_train_loss / train_size
        train_acc = epoch_train_corrects / train_size
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_corrects = 0.0, 0

        with torch.no_grad():

            # Validation batch loop
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1)
                epoch_val_corrects += (preds == labels).sum().item()

        # Average validation loss and accuracy for the epoch
        val_loss = epoch_val_loss / val_size
        val_acc = epoch_val_corrects / val_size
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Scheduler step based on validation loss
        scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Saving best weights and plotting
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved to: {save_path}")

    epochs_range = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("efficientnetb3_training_curves.png")
    plt.show()