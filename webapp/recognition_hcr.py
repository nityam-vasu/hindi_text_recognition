import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the Model
class HindiOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(HindiOCRModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load Reference
def load_reference(ref_file):
    ref = np.load(ref_file, allow_pickle=True).item()
    return ref

# Image Preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

# Load Model
def load_model(model_path, num_classes, device):
    model = HindiOCRModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict Function
def predict(image_path, model, transform, device, ref):
    image = Image.open(image_path)
    preprocessed_image = transform(image)
    print("Input shape to model:", preprocessed_image.unsqueeze(0).shape)
    preprocessed_pil = transforms.ToPILImage()(preprocessed_image)
    #preprocessed_pil.save("preprocessed_unknown_image.png")
    image = preprocessed_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        predicted_idx = outputs.argmax(dim=1).item()
        prediction = ref[predicted_idx] if ref else str(predicted_idx)
        print("Predicted index:", predicted_idx)
    
    return prediction
