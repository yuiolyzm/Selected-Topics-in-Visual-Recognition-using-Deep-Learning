import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import pandas as pd
from torchvision.models import resnext50_32x4d
from PIL import Image

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

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def create_prediction_mapping():
    mapping = {0: 0, 1: 1}
    for original in range(2, 100):
        base = (original - 1) // 11
        offset = original - (base * 11 + 1)
        if offset == 0:
            mapping[original] = base + 1
        else:
            mapping[original] = (base + 1) * 10 + offset - 1
    return mapping

def load_model(model_path, num_classes=100):
    model = resnext50_32x4d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def test_and_save_predictions(model, test_loader, device, mapping):
    model.to(device)
    model.eval()
    predictions = []

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transform = get_test_transform()
    test_dataset = TestDataset("./data/test", transform=test_transform)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model(r"C:\Users\user\Downloads\best_model\best_model.pth")
    mapping = create_prediction_mapping()
    test_and_save_predictions(model, test_loader, device, mapping)

if __name__ == "__main__":
    main()
