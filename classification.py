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
        self.model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
        self.model.eval()
        self.create_widgets()
        self.center_window()
        root.resizable(width=False, height=False)

    def create_widgets(self):
        widget_style = {
            'font': ("Helvetica", 12),
            'bg': '#f0f0f0',
            'fg': '#333',
        }

        # Result Label
        self.result_label = tk.Label(self.master, text="Classification Result: ", **widget_style)
        self.result_label.pack(pady=20)

        self.image_label = tk.Label(self.master, borderwidth=2, relief="solid", **widget_style)

        self.image_label.pack(pady=10, padx=10)

        self.select_button = tk.Button(self.master, text="Select Image", command=self.load_image, **widget_style)
        self.select_button.config(
            borderwidth=0,
            highlightthickness=0,
            bd=0,
            bg="#4CAF50",
            fg="white",
        )
        self.select_button.pack(side='bottom', pady=30, ipadx=15, ipady=8)  # Increased padding, button size

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.png")])
        if not file_path:
            return
        image = Image.open(file_path)
        image = image.resize((image_size, image_size))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        result = self.classify_image(file_path)
        self.result_label.config(text=f"Classification Result: {result}")

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

        class_folders = [folder for folder in os.listdir(data_path_main) if
                         os.path.isdir(os.path.join(data_path_main, folder))]

        if 0 <= predicted_class_idx < len(class_folders):
            predicted_class_name = class_folders[predicted_class_idx]
        else:
            predicted_class_name = "Unknown Class"

        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence_scores = [f"{class_folders[i]}: {probabilities[i]:.2%}" for i in range(len(class_folders))]

        result_text = f"{predicted_class_name}\n\n" + "\n".join(confidence_scores)
        return result_text

    def center_window(self):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        preferred_width = 360
        preferred_height = 500
        if preferred_width > screen_width * 0.8:
            preferred_width = int(screen_width * 0.8)
        if preferred_height > screen_height * 0.8:
            preferred_height = int(screen_height * 0.8)
        x = (screen_width - preferred_width) // 2
        y = (screen_height - preferred_height) // 2
        self.master.geometry(f"{preferred_width}x{preferred_height}+{x}+{y}")
        self.master.resizable(True, True)
        self.master.minsize(360, 500)
        self.master.maxsize(screen_width, screen_height)


if __name__ == "__main__":
    root = tk.Tk()
    root.wm_attributes('-toolwindow', 'True')
    app = ImageClassifierGUI(root)
    root.mainloop()
