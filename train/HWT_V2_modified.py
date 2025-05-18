#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
import torchvision.transforms.functional as TF
import argparse
import datetime

# Custom Transform to Resize and Pad
def resize_and_pad(image, target_height=36, target_width=128):
    # Convert PIL image to tensor to get dimensions
    img_tensor = transforms.ToTensor()(image)
    _, h, w = img_tensor.shape

    # Resize to target height while preserving aspect ratio
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    image = transforms.Resize((target_height, new_width))(image)

    # Convert back to tensor to get new dimensions
    img_tensor = transforms.ToTensor()(image)
    _, _, new_w = img_tensor.shape

    # Pad to target width
    padding_left = (target_width - new_w) // 2
    padding_right = target_width - new_w - padding_left
    padding = (padding_left, 0, padding_right, 0)  # (left, top, right, bottom)
    image = TF.pad(image, padding, fill=255)  # Pad with white (255 for grayscale)

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
        x = x.squeeze(2)  # Should now be (batch_size, channels, width)
        x = x.permute(0, 2, 1)  # (batch_size, width, channels)
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

# Custom Dataset
class HindiDataset(Dataset):
    def __init__(self, txt_file, root_dir, char_to_idx, transform=None):
        self.data = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                path, label = line.strip().split(maxsplit=1)
                self.data.append((path, label))
        self.root_dir = root_dir
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.missing_chars = set()
        for _, label in self.data:
            for c in label:
                if c not in self.char_to_idx:
                    self.missing_chars.add(c)
        if self.missing_chars:
            print(f"Warning: Characters not in charlist.txt: {self.missing_chars}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('L')
        if self.transform:
            img = self.transform(img)
        label_indices = [self.char_to_idx[c] for c in label if c in self.char_to_idx]
        return img, label_indices

# Collate Function
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels

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

# Function to Save Model
def save_model(model, accuracy, best_accuracy, save_path):
    os.makedirs(save_path, exist_ok=True)
    today_date = datetime.date.today().strftime("%Y%m%d")
    model_name = f"HWT_recognition_model_{today_date}.pth"
    full_path = os.path.join(save_path, model_name)
    if accuracy > best_accuracy:
        torch.save(model.state_dict(), full_path)
        print(f"New best accuracy {accuracy:.4f} > {best_accuracy:.4f}. Model saved to {full_path}")
        return accuracy
    else:
        print(f"Accuracy {accuracy:.4f} <= {best_accuracy:.4f}. Model not saved.")
        return best_accuracy

# Data Augmentation with Aspect Ratio Preservation
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_and_pad(img, target_height=36, target_width=128)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_and_pad(img, target_height=36, target_width=128)),
    transforms.ToTensor(),
])

# Main Function
def main(args):
    # Configuration
    root_dir = 'HindiSeg/'
    train_txt = os.path.join(root_dir, 'train.txt')
    val_txt = os.path.join(root_dir, 'val.txt')
    charlist_file = os.path.join(root_dir, 'charlist.txt')
    num_classes = 112  # Without Blanks (with blanks 113)
    batch_size = 32
    num_epochs = args.epochs
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Character List
    with open(charlist_file, 'r', encoding='utf-8') as f:
        charlist_str = f.read().strip()
    charlist = list(charlist_str)
    print(f"Loaded {len(charlist)} characters from charlist.txt")
    if len(charlist) != num_classes:
        print(f"Warning: Expected {num_classes} characters, but loaded {len(charlist)}")

    # Create char_to_idx using len
    char_to_idx = {}
    for char in charlist:
        char_to_idx[char] = len(char_to_idx) + 1
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Data Loaders
    train_dataset = HindiDataset(train_txt, root_dir, char_to_idx, train_transform)
    val_dataset = HindiDataset(val_txt, root_dir, char_to_idx, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, Loss, and Optimizer
    model = CRNN(num_classes + 1).to(device)  # Add 1 for blank label
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early Stopping
    patience = args.patience
    best_val_loss = float('inf')
    counter = 0
    best_accuracy = -float('inf')

    # Training Loop with tqdm
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            log_probs = F.log_softmax(outputs, dim=2)
            targets = torch.cat([torch.tensor(label, dtype=torch.long) for label in labels]).to(device)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # Validation Loop with tqdm
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_bar):
                images = images.to(device)
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                log_probs = F.log_softmax(outputs, dim=2)
                targets = torch.cat([torch.tensor(label, dtype=torch.long) for label in labels]).to(device)
                target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
                input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long).to(device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                preds = greedy_decode(log_probs, idx_to_char)
                all_preds.extend(preds)
                all_labels.extend(labels)
                for pred, label in zip(preds, labels):
                    label_str = ''.join([idx_to_char[idx] for idx in label])
                    if pred == label_str:
                        correct += 1
                    total += 1
                val_bar.set_postfix({'accuracy': correct / total})
        accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Accuracy: {accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Random Sample Printing After Validation
        print(f"\nRandom Samples After Epoch {epoch+1} Validation:")
        random_indices = random.sample(range(len(all_preds)), min(3, len(all_preds)))
        for idx, sample_idx in enumerate(random_indices):
            pred = all_preds[sample_idx]
            label = all_labels[sample_idx]
            label_str = ''.join([idx_to_char[idx] for idx in label])
            print(f"Random Sample {idx+1} - Predicted: {pred}, Ground Truth: {label_str}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_accuracy = save_model(model, accuracy, best_accuracy, args.save_path)
        else:
            counter += 1
            print(f"Validation loss did not improve for {counter} epochs.")
            if counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CRNN model for Handwritten Hindi Word Recognition (HWT).')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model (default: 50)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--save_path', type=str, default='webapp/essential/models/', help='Directory to save the trained model (default: webapp/essential/models/)')
    args = parser.parse_args()
    main(args)