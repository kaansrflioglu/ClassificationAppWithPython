import os
import cv2
from keras_preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)
img_size = config['input-size']


def augment_data(input_dir, output_dir, augmentation_factor=5, min_size=50, target_size=(img_size, img_size)):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_folder)
        output_class_path = os.path.join(output_dir, f"{class_folder}_augmented")

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        image_files = os.listdir(class_path)

        for i, image_file in enumerate(tqdm(image_files, desc=f"Class {class_folder}")):
            img_path = os.path.join(class_path, image_file)

            try:
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    print(f"Warning: Image could not be read or is empty: {img_path}")
                    continue

                if img.shape[0] < min_size or img.shape[1] < min_size:
                    print(f"Warning: Image size too small: {img_path}")
                    continue

                img = cv2.resize(img, target_size)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.reshape((1,) + img.shape)

                filename, _ = os.path.splitext(image_file)
                save_prefix = filename

                j = 0
                for _ in datagen.flow(
                        img, batch_size=1,
                        save_to_dir=output_class_path,
                        save_prefix=save_prefix,
                        save_format='jpg'
                ):
                    j += 1
                    if j >= augmentation_factor:
                        break
            except Exception as e:
                print(f"Error: An error occurred while processing the image: {img_path}")
                print(e)


input_directory = config["data_path_main"]
output_directory = input_directory + "_augmented"
augment_data(input_directory, output_directory)
