import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class FERPlusDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        emotions = self.data.iloc[idx, 3:11].astype(int).values
        label = int(emotions.argmax())
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = loss_sum / len(dataloader)
    print(f"Validation Accuracy: {acc:.2f}%  |  Avg Loss: {avg_loss:.4f}")
    return y_true, y_pred

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_dataset = FERPlusDataset("label_valid.csv", "FER2013Valid", transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 8)
    model.load_state_dict(torch.load("mobilenetv2_ferplus.pth", map_location=device))
    model = model.to(device)

    y_true, y_pred = evaluate(model, val_loader, device)

    class_names = ['neutral', 'happiness', 'surprise', 'sadness',
                   'anger', 'disgust', 'fear', 'contempt']
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("FER+ Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png") 
    print("picture saved as confusion_matrix.png")
