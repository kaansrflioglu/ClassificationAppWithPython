import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import SimpleCNN
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

image_size = config['input-size']
data_path_main = config['data_path_main']


class ImageClassifierGUI:
    def __init__(self, master):
        self.result_label = None
        self.image_label = None
        self.select_button = None
        self.master = master
        self.master.title("Image Classifier")
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()
        self.create_widgets()
        self.center_window()
        root.resizable(width=False, height=False)

    def create_widgets(self):
        self.result_label = tk.Label(self.master, text="Sınıflandırma Sonucu: ")
        self.result_label.pack(pady=16)
        self.image_label = tk.Label(self.master, borderwidth=2, relief="solid")
        self.image_label.pack()
        self.select_button = tk.Button(self.master, text="Resim Seç", command=self.load_image,
                                       borderwidth=2, relief="groove")
        self.select_button.pack(side='bottom', pady=26)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Resim Seç", filetypes=[("Image files", "*.jpg;*.png")])
        if not file_path:
            return
        image = Image.open(file_path)
        image = image.resize((image_size, image_size))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        result = self.classify_image(file_path)
        self.result_label.config(text=f"Sınıflandırma Sonucu: {result}")

    def classify_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = transform(Image.open(image_path).convert("RGB"))
        input_image = input_image.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_image)

        _, predicted_class = output.max(1)
        predicted_class_idx = predicted_class.item()

        # Get the list of folder names in data_path_main
        class_folders = [folder for folder in os.listdir(data_path_main) if
                         os.path.isdir(os.path.join(data_path_main, folder))]

        if 0 <= predicted_class_idx < len(class_folders):
            predicted_class_name = class_folders[predicted_class_idx]
        else:
            predicted_class_name = "Unknown Class"

        return predicted_class_name

    def center_window(self):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"280x360+"
                             f"{(screen_width - 280) // 2}+"
                             f"{(screen_height - 360) // 2}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()
