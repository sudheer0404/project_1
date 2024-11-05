import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from dehazetransformer import DehazeFormer  # Adjust if necessary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DehazeFormer().to(device)
model.load_state_dict(torch.load('dehaze_model_better.pth', map_location=device))
model.eval()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
def dehaze_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_tensor = torch.clamp(output_tensor.squeeze(0).cpu(), 0, 1)
    output_image = transforms.ToPILImage()(output_tensor)
    return output_image
hazy_image_path = "hellohaze.webp"
dehazed_image = dehaze_image(hazy_image_path)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(Image.open(hazy_image_path))
axes[0].set_title("Hazy Image")
axes[0].axis("off")
axes[1].imshow(dehazed_image)
axes[1].set_title("Dehazed Image")
axes[1].axis("off")
plt.show()
