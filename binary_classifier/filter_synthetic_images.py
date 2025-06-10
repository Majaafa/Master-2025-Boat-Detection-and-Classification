# Classifies all images in a folder as 'good' or 'bad' using trained EfficientNet-B3 and saves them to separate folders

import os
import torch
from torchvision import transforms
from PIL import Image
import timm
import shutil

# Trained model path
model_path = "path/to/classifier_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resize and normalize (same as training, no augmentation)
transform = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model with same structure as during training, pytorch requires same architecture as in training
model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=2)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.classifier.in_features, 2)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()  # disables dropout and batchnorm randomness

# Input and output folders
input_folder = "path/to/your/input/images"
output_folder = "classified_output"
os.makedirs(os.path.join(output_folder, "good"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "bad"), exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in image_files:
    path = os.path.join(input_folder, filename)
    image = Image.open(path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    label = "good" if pred == 0 else "bad"
    dest_path = os.path.join(output_folder, label, filename)
    shutil.copy(path, dest_path)  # save image to predicted class folder

    print(f"{filename}: {label}")
