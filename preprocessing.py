import os
import torch
from torchvision import transforms
from PIL import Image

from main import data_path_main


class ImagePreprocessor:
    def __init__(self, input_size=(224, 224), contrast_factor=1.5):
        self.input_size = input_size
        self.contrast_factor = contrast_factor

    def apply_transforms(self, image):
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=self.contrast_factor),
            transforms.ToTensor(),
        ])
        return transform(image)


def load_and_preprocess_image(image_path, preprocessor):
    image = Image.open(image_path).convert("RGB")
    preprocessed_image = preprocessor.apply_transforms(image)
    preprocessed_image = torch.unsqueeze(preprocessed_image, 0)
    return preprocessed_image


def main(data_path):
    data_path = data_path_main
    preprocessor = ImagePreprocessor()

    image_path = os.path.join(data_path, "path_to_an_image.jpg")
    preprocessed_image = load_and_preprocess_image(image_path, preprocessor)

    print("Preprocessed image shape:", preprocessed_image.shape)


if __name__ == "__main__":
    main(data_path_main)
