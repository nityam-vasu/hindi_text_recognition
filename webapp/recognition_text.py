import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

# Custom Transform to Resize and Pad
def resize_and_pad(image, target_height=36, target_width=128):
    img_tensor = transforms.ToTensor()(image)
    _, h, w = img_tensor.shape
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    image = transforms.Resize((target_height, new_width))(image)
    img_tensor = transforms.ToTensor()(image)
    _, _, new_w = img_tensor.shape
    padding_left = (target_width - new_w) // 2
    padding_right = target_width - new_w - padding_left
    padding = (padding_left, 0, padding_right, 0)
    image = TF.pad(image, padding, fill=255)
    return image

# CNN Component
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, (9, 3), padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        return x

# BLSTM Component
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

# CRNN Model with Dropout
class CRNN(nn.Module):
    def __init__(self, num_classes, cnn_output_size=256, lstm_hidden_size=512):
        super(CRNN, self).__init__()
        self.cnn = CNN()
        self.rnn = BLSTM(cnn_output_size, lstm_hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

# CTC Greedy Decoder
def greedy_decode(output, idx_to_char, blank_label=0):
    arg_max = output.argmax(dim=2).transpose(0, 1)
    decodes = []
    for seq in arg_max:
        decode = []
        previous = None
        for idx in seq:
            idx = idx.item()
            if idx != blank_label and idx != previous:
                if idx in idx_to_char:
                    decode.append(idx_to_char[idx])
            previous = idx
        decodes.append(''.join(decode))
    return decodes

# Load Character List
def load_charlist(charlist_file):
    with open(charlist_file, 'r', encoding='utf-8') as f:
        charlist_str = f.read().strip()
    charlist = list(charlist_str)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charlist)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

# Image Preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: resize_and_pad(img, target_height=36, target_width=128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

# Load Model
def load_model(model_path, num_classes, device):
    model = CRNN(num_classes, cnn_output_size=256, lstm_hidden_size=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict Function
def predict(image_path, model, idx_to_char, transform, device):
    image = Image.open(image_path)
    preprocessed_image = transform(image)
    print("Input shape to model:", preprocessed_image.unsqueeze(0).shape)
    preprocessed_pil = transforms.ToPILImage()(preprocessed_image)
    #preprocessed_pil.save("preprocessed_unknown_image.png")
    image = preprocessed_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.permute(1, 0, 2)
        log_probs = F.log_softmax(outputs, dim=2)
        arg_max = outputs.argmax(dim=2).transpose(0, 1)
        top_probs = F.softmax(outputs, dim=2).max(dim=2)[0].transpose(0, 1)
        print("Raw predicted indices:", arg_max[0].tolist())
        print("Top probabilities:", top_probs[0].tolist())
        preds = greedy_decode(log_probs, idx_to_char)
        prediction = preds[0] if preds else ""
    
    return prediction
