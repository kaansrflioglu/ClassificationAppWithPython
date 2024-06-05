import os
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

import cv2
import numpy as np


def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=os.path.expanduser("~"),
        title="Select Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("all files", "*.*"))
    )
    root.destroy()

    if not file_path:
        print("No image selected.")
        return None

    return file_path


def process_image(path, output_dir="layers"):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Error loading image at {path}!")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    turkuaz_lower = np.array([150, 50, 50])
    turkuaz_upper = np.array([170, 255, 255])
    yesil_lower = np.array([40, 50, 50])
    yesil_upper = np.array([90, 255, 255])

    turkuaz_mask = cv2.inRange(hsv_image, turkuaz_lower, turkuaz_upper)
    yesil_mask = cv2.inRange(hsv_image, yesil_lower, yesil_upper)
    combined_mask = turkuaz_mask | yesil_mask

    canny_edges = cv2.Canny(combined_mask, 30, 150)
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%m-%S")

    os.makedirs(output_dir, exist_ok=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:
            m = cv2.moments(contour)
            if m["m00"] != 0:
                c_x = int(m["m10"] / m["m00"])
                c_y = int(m["m01"] / m["m00"])
                cv2.circle(image, (c_x, c_y), 5, (0, 255, 0), 2)

        else:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)

    save_path = os.path.join(output_dir, f"layer_boundaries_{timestamp}.png")
    success = cv2.imwrite(save_path, image)
    if success:
        print(f"Image successfully saved to {save_path}")
    else:
        raise IOError(f"Error saving image to {save_path}!")


if __name__ == "__main__":
    image_path = select_image()
    if image_path:
        try:
            process_image(image_path)
        except (FileNotFoundError, IOError) as e:
            print(e)
