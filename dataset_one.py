import numpy as np
import os
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt


# Paths for the numpy array folders
npy_folder = './carseg_data/arrays'
cropped_npy_folder = './carseg_data/dataset_one_arrays'

# Paths for the image folders
images_folder = './carseg_data/all_images'
cropped_images_folder = './carseg_data/dataset_one_images'

# Function to crop and resize the image
def crop_and_resize(image, target_width, target_height):
    # Calculate the smaller dimension and crop to a square
    smaller_side = min(image.size)
    left = (image.width - smaller_side) / 2
    top = (image.height - smaller_side) / 2
    image = image.crop((left, top, left + smaller_side, top + smaller_side))
    return image.resize((target_width, target_height), Image.LANCZOS)

# Function to crop and resize segmentation data
def crop_and_resize_segmentation(data, target_width, target_height):
    # Calculate the smaller dimension and crop to a square
    smaller_side = min(data.shape[0], data.shape[1])
    top = (data.shape[0] - smaller_side) // 2
    left = (data.shape[1] - smaller_side) // 2
    data = data[top:top+smaller_side, left:left+smaller_side]
    return np.array(Image.fromarray(data).resize((target_width, target_height)))

example_shown = False

# Ensure the output directory exists
os.makedirs(cropped_images_folder, exist_ok=True)

# Process all image files
for image_file in os.listdir(images_folder):
    if image_file.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(images_folder, image_file)
        image = Image.open(image_path)

        # Display the original image
        if not example_shown:
            plt.imshow(image)
            plt.title("Original Image")
            plt.show()

        # Crop and resize the image
        processed_image = crop_and_resize(image, 256, 256)

        # Display the processed image
        if not example_shown:
            plt.imshow(processed_image)
            plt.title("Cropped and Resized Image")
            plt.show()
            example_shown = True  # Set the flag to True after showing the example

        # Save the processed image
        processed_image.save(os.path.join(cropped_images_folder, image_file))

# Ensure the output directory for cropped arrays exists
os.makedirs(cropped_npy_folder, exist_ok=True)

# Process all .npy files
for npy_file in os.listdir(npy_folder):
    if npy_file.lower().endswith('.npy'):
        npy_path = os.path.join(npy_folder, npy_file)
        segmentation_data = np.load(npy_path)

        # Crop and resize the segmentation data
        processed_data = crop_and_resize_segmentation(segmentation_data, 256, 256)

        # Save the processed data
        np.save(os.path.join(cropped_npy_folder, npy_file), processed_data)


