import os #training transformer
import torch
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dehazetransformer import DehazeFormer
class HazyClearDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(hazy_dir))
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.image_filenames[idx])
        clear_path = os.path.join(self.clear_dir, self.image_filenames[idx])       
        hazy_image = Image.open(hazy_path).convert("RGB")
        clear_image = Image.open(clear_path).convert("RGB")
        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)       
        return hazy_image, clear_image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
train_dataset = HazyClearDataset('preprocessed_data/train/hazy', 'preprocessed_data/train/clear', transform=transform)
val_dataset = HazyClearDataset('preprocessed_data/test_thin/hazy', 'preprocessed_data/test_thin/clear', transform=transform)
test_dataset = HazyClearDataset('preprocessed_data/test_moderate/hazy', 'preprocessed_data/test_moderate/clear', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DehazeFormer().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}...")
    model.train()
    running_loss = 0.0
    for batch_idx, (hazy_images, clear_images) in enumerate(train_loader):
        hazy_images = hazy_images.to(device)
        clear_images = clear_images.to(device)        
        optimizer.zero_grad()
        outputs = model(hazy_images) #forward pass
        loss = criterion(outputs, clear_images)
        loss.backward() #backward pass
        optimizer.step()
        running_loss += loss.item()     
        print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
    model.eval() #validation
    val_loss = 0.0
    with torch.no_grad():
        for hazy_images, clear_images in val_loader:
            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)
            outputs = model(hazy_images)
            loss = criterion(outputs, clear_images)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(val_loader)}")

torch.save(model.state_dict(), 'dehaze_model_better.pth')
print("Model saved as dehaze_model_better.pth")
