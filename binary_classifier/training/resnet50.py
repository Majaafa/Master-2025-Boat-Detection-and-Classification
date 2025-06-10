# Trained with 1000 good and 1000 bad synthetic images

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import numpy as np
import copy

# Windows requires __name__ == "__main__" loop because of multiprocessing issues
if __name__ == "__main__":
    # Setup
    data_dir = r"c:\Users\halvo\Documents\Boat-Classification-Master\good_and_bad_synthetic"
    batch_size = 16
    epochs = 25
    val_ratio = 0.2
    early_stopping = 4
    learning_rate = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experimental values for augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(13)
        ], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.2)
        ], p=0.4),
        transforms.RandomApply([
            transforms.GaussianBlur(3, sigma=(0.1, 2.0))
        ], p=0.25),
        transforms.ToTensor(),
        # ImageNet normalization values (standard for pre-trained models)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset and dataloader logic
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    # Load pre-trained resnet50 ImageNet weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freezing can be experimental, based on dataset size and complexity
    # https://medium.com/we-talk-data/guide-to-freezing-layers-in-pytorch-best-practices-and-practical-examples-8e644e7a9598
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    # Dropout rate also experimental, usually between 0.2 and 0.5. Added here to final layer
    # https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )

    # Move model to cuda if available
    model = model.to(device)

    # Object function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # Learning rate scheduler to reduce learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Early stopping parameters and intialization
    train_loss, val_loss = [], []
    train_accuracy, val_accuracy = [], []

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
        train_epoch_loss = epoch_train_loss / train_size
        train_epoch_acc = epoch_train_corrects / train_size
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_acc)

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
        val_epoch_loss = epoch_val_loss / val_size
        val_epoch_acc = epoch_val_corrects / val_size
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_acc)

        # Scheduler step based on validation loss
        scheduler.step(val_epoch_loss)

        # Early stopping
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_epoch_loss:.4f}, Acc: {train_epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")

    # Saving best weights and plotting
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "resnet50_best_good_bad_classifier2.pth")
    print("Best model saved as resnet50_best_good_bad_classifier.pth")

    epochs_range = np.arange(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracy, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("resnet50_training_curves.png")
    plt.show()
