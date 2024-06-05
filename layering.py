import os
from datetime import datetime
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    root.destroy()

    if not file_path:
        print("No image selected.")
        return None

    return file_path


selected_image_path = select_image()

if selected_image_path:
    image = cv2.imread(selected_image_path)

    height, width, _ = image.shape

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    points = []
    for y in range(height):
        for x in range(width):
            if np.array_equal(image[y, x], [0, 255, 255]):
                points.append((x, y))

    row_points = [[] for _ in range(height)]
    for x, y in points:
        row_points[y].append(x)

    zigzag_points = []
    for y, points in enumerate(row_points):
        if points:
            three_quarter_index = int(len(points) * 3 / 4)
            three_quarter_x = points[three_quarter_index] if three_quarter_index < len(points) else points[-1]
            zigzag_points.append((three_quarter_x, y))

    os.makedirs("layers", exist_ok=True)

    for i in range(1, len(zigzag_points)):
        cv2.line(image, zigzag_points[i - 1], zigzag_points[i], (0, 0, 0), 5)

    save_path = os.path.join("layers/", f"output_{timestamp}.png")
    success = cv2.imwrite(save_path, image)
    if not success:
        print(f"Error saving image to {save_path}!")
    else:
        print(f"Image successfully saved to {save_path}")
else:
    pass
