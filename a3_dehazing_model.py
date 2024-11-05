import torch
from torchvision import transforms
from PIL import Image
from dehazetransformer import DehazeFormer

class Dehazer:
    def __init__(self, model_path='dehaze_model_better.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DehazeFormer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor without resizing
        ])

    def dehaze_image(self, image_path):
        # Load the image and keep the original size
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Apply transformation and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Run the model prediction
            output_tensor = self.model(input_tensor)
        
        # Clamp values and convert back to an image
        output_tensor = torch.clamp(output_tensor.squeeze(0).cpu(), 0, 1)
        output_image = transforms.ToPILImage()(output_tensor)
        
        # Resize output to the original size (just in case the model output differs)
        output_image = output_image.resize(original_size)
        return output_image

# Example usage:
# dehazer = Dehazer(model_path='dehaze_model_better.pth')
# dehazed_image = dehazer.dehaze_image('hellohaze.webp')
# dehazed_image.show()
