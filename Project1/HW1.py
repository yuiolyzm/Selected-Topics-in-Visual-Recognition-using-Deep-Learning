import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torchvision.models import resnext50_32x4d
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset for Test Images
class TestDataset(data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(400),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

def get_dataloaders(train_transform, val_test_transform, batch_size=128):
    train_dir = os.path.join("./data", 'train')
    val_dir = os.path.join("./data", 'val')
    test_dir = os.path.join("./data", 'test')

    dataset_train = datasets.ImageFolder(train_dir, transform=train_transform)
    dataset_val = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = TestDataset(test_dir, transform=val_test_transform)

    train_loader = data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=32
    )
    val_loader = data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=32
    )
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

def get_model(num_classes=100):
    model = resnext50_32x4d(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def compute_class_weights(dataset):
    labels = [label for _, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    return class_weights / class_weights.sum()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # if self.alpha is not None:
        #     focal_loss *= self.alpha[targets]

        # if self.reduction == "mean":
        #     return focal_loss.mean()
        # elif self.reduction == "sum":
        #     return focal_loss.sum()
        return focal_loss.mean()

def get_loss_function(train_dataset, use_focal_loss=False):
    class_weights = compute_class_weights(train_dataset).to(device)
    if use_focal_loss:
        return FocalLoss(alpha=1.0, gamma=2.0)
    return nn.CrossEntropyLoss(weight=class_weights)

def plot_training_curve(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve_129.png")  
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(val_accuracies, label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve_129.png")

def train_and_validate(model, train_loader, val_loader, device, lr=0.0007, 
                       num_epochs=50, patience=7, use_focal_loss=False):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.4, patience=5)

    train_dataset = train_loader.dataset
    # criterion = get_loss_function(train_dataset, use_focal_loss=use_focal_loss)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float("inf")
    early_stop_counter = 0
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        if epoch == 3:
            for param in model.parameters():
                param.requires_grad = True

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                correct += (torch.argmax(outputs, 1) == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = correct / total
        val_accuracies.append(accuracy) 
        print(
            f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Epoch {epoch+1}: New best model saved.")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}, "
              f"lr: {', '.join([f'{lr:.6f}' for lr in scheduler.get_last_lr()])}"
        )

    plot_training_curve(train_losses, val_losses, val_accuracies)

def create_prediction_mapping():
    mapping = {0: 0, 1: 1}
    for original in range(2, 100):
        base = (original - 1) // 11  # Find base group (0-9)
        offset = original - (base * 11 + 1)  # Determine new range start
        if offset == 0:
            mapping[original] = base + 1
        else:
            mapping[original] = (base + 1) * 10 + offset - 1
    return mapping

def test_and_save_predictions(model, test_loader, device, mapping):
    predictions = []
    model.eval()
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            mapped_preds = [mapping[p.item()] for p in preds.cpu()]
            img_names = [name.replace('.jpg', '') for name in img_names]
            predictions.extend(zip(img_names, mapped_preds))
    
    df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
    df.to_csv("prediction.csv", index=False)
    print("Test predictions saved to prediction.csv")

def main():
    train_transform, val_test_transform = get_transforms()
    train_loader, val_loader, test_loader = get_dataloaders(train_transform, 
                                                            val_test_transform, batch_size=128)
    model = get_model(num_classes=100)
    train_and_validate(
        model, train_loader, val_loader, device, 
        lr=0.0007, num_epochs=50, patience=10, use_focal_loss=True
    )

    mapping = create_prediction_mapping()
    test_and_save_predictions(model, test_loader, device, mapping)    

if __name__ == "__main__":
    main()
