import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from dehazetransformer import DehazeFormer  # Adjust if necessary

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and set it to evaluation mode
model = DehazeFormer().to(device)
model.load_state_dict(torch.load('dehaze_model_better.pth', map_location=device))
model.eval()

# Define transformations without resizing
transform = transforms.Compose([
    transforms.ToTensor(),
])

def dehaze_image(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run the model prediction
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Clamp and resize output to the original size
    output_tensor = torch.clamp(output_tensor.squeeze(0).cpu(), 0, 1)
    output_image = transforms.ToPILImage()(output_tensor)
    output_image = output_image.resize(original_size)  # Restore original size
    return output_image

# Path to hazy image
hazy_image_path = "hellohaze.webp"
dehazed_image = dehaze_image(hazy_image_path)

# Display the original and dehazed images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(Image.open(hazy_image_path))
axes[0].set_title("Hazy Image")
axes[0].axis("off")
axes[1].imshow(dehazed_image)
axes[1].set_title("Dehazed Image")
axes[1].axis("off")
plt.show()
