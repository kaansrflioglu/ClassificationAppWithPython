import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
from datetime import datetime


def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    root.destroy()
    return file_path


selected_image_path = select_image()

if selected_image_path:
    image = cv2.imread(selected_image_path)
    if image is None:
        print("Error loading image!")
    else:
        height, width, _ = image.shape

        cv2.line(image, (0, height // 2), (width, height // 2), (255, 255, 255), 2)
        cv2.line(image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

        num_lines = 10
        line_spacing = height // (num_lines + 1)
        blue_pixels = np.where(np.all(image == [255, 0, 0], axis=-1))
        point_counts = []
        dot_color = (0, 255, 255)

        for i in range(1, num_lines + 1):
            y = i * line_spacing
            cv2.line(image, (0, y), (width, y), (255, 255, 255), 1)

            intersect_points = [(x, y) for x, y in zip(blue_pixels[1], blue_pixels[0]) if y == i * line_spacing]
            for point in intersect_points:
                cv2.circle(image, point, 2, dot_color, 2)
            point_counts.append(len(intersect_points))

        if point_counts:
            median_count = np.median(point_counts)
            median_line_index = np.argsort(point_counts)[len(point_counts) // 2]
            median_y = (median_line_index + 1) * line_spacing
            if median_count > 0:
                cv2.circle(image, (width // 2, median_y), 5, (147, 20, 255), -1)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join("layers/", f"layer_dots_{timestamp}.png")
            success = cv2.imwrite(save_path, image)
            if not success:
                print(f"Error saving image to {save_path}!")
            else:
                print(f"Image successfully saved to {save_path}")
        else:
            print("No blue pixels found.")

else:
    print("No image selected.")
