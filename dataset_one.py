import os
import shutil
import numpy as np
from PIL import Image
import random
import matplotlib.colors as mcolors

# Paths to folders
input_folder = './carseg_data/arrays'
landscapes_folder = './carseg_data/cropped_landscapes'
output_folder_combined = './carseg_data/combined_images'

# Create output folders if they do not exist
os.makedirs(output_folder_combined, exist_ok=True)

def adjust_hue(image, hue_change):
    # Convert the image to HSV format and modify the hue
    hsv_image = mcolors.rgb_to_hsv(image/255.0)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_change) % 1.0
    return (mcolors.hsv_to_rgb(hsv_image) * 255).astype(np.uint8)

def create_mask_with_segments(image_data, segmentation_map):
    # Set all non-zero values in the segmentation map to 255
    segmentation_map = np.where(segmentation_map != 0, 255, segmentation_map)

    # Create mask with segments
    mask_with_segments = np.zeros((image_data.shape[0], image_data.shape[1], 4), dtype=np.uint8)
    
    # Assign RGB values from the original image
    mask_with_segments[..., :3] = image_data

    # Assign modified segmentation map to the fourth channel
    mask_with_segments[..., 3] = segmentation_map

    return mask_with_segments

def combine_with_random_background(mask_image_pil, landscapes_folder):
    # Combine with a random background
    landscape_images = [f for f in os.listdir(landscapes_folder) if f.endswith(('.png', '.jpg'))]
    landscape_image = Image.open(os.path.join(landscapes_folder, random.choice(landscape_images)))
    landscape_image = landscape_image.resize(mask_image_pil.size)

    # Ensure the background image is in RGB format
    if landscape_image.mode != 'RGB':
        landscape_image = landscape_image.convert('RGB')

    # Use the fourth channel (alpha) of the mask for blending
    combined_image = Image.composite(mask_image_pil, landscape_image, mask_image_pil.split()[3])

    # Ensure the combined image is in RGB format
    if combined_image.mode != 'RGB':
        combined_image = combined_image.convert('RGB')

    return combined_image

# List of hue changes
hue_changes = np.linspace(0, 1, 20, endpoint=False)  # 20 different colors
hue_index = 0

for file_name in os.listdir(input_folder):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_folder, file_name)
        data = np.load(file_path)

        image_data, segmentation_map = data[..., :3], data[..., 3]

        # Modify hue of segments using the next value from the list
        hue_change = hue_changes[hue_index]
        image_data = adjust_hue(image_data, hue_change)

        # Create mask with segments
        mask_with_segments = create_mask_with_segments(image_data, segmentation_map)

        # Convert to PIL format and combine with background
        mask_image_pil = Image.fromarray(mask_with_segments)
        combined_image = combine_with_random_background(mask_image_pil, landscapes_folder)
        combined_image.save(os.path.join(output_folder_combined, file_name.replace('.npy', '.png')))

        # Update hue change index
        hue_index = (hue_index + 1) % len(hue_changes)

### PREPARE THE DATASET
training_one = './carseg_data/training_one'

def copy_folder(src, dst):
    if os.path.exists(dst):
        print("The output folder exists.")
    else:
        # Copy the folder with all its content
        shutil.copytree(src, dst)
        print(f"The folder '{src}' was copied to location '{dst}'")

def prepare_dataset(images_folder, input_folder, output_folder):
    # Paths for the output subfolders
    train_img_path = os.path.join(output_folder, 'train', 'img')
    train_arrays_path = os.path.join(output_folder, 'train', 'arrays')
    val_img_path = os.path.join(output_folder, 'val', 'img')
    val_arrays_path = os.path.join(output_folder, 'val', 'arrays')

    # Create necessary directories
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_arrays_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(val_arrays_path, exist_ok=True)

    # Get all image and mask filenames
    images = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg'))]
    masks = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    # Filter out the common filenames
    common_filenames = set([os.path.splitext(f)[0] for f in images]) & set([os.path.splitext(f)[0] for f in masks])

    # Function to check and copy the corresponding image file
    def copy_image_file(src_folder, dest_folder, filename):
        for ext in ['.jpg', '.png']:
            if os.path.exists(os.path.join(src_folder, filename + ext)):
                shutil.copy(os.path.join(src_folder, filename + ext), dest_folder)
                break

    # Split data into training and validation sets (5% for validation)
    val_filenames = set(np.random.choice(list(common_filenames), size=int(len(common_filenames) * 0.05), replace=False))
    train_filenames = common_filenames - val_filenames

    # Copy files to their respective directories
    for filename in train_filenames:
        copy_image_file(images_folder, train_img_path, filename)
        shutil.copy(os.path.join(input_folder, filename + '.npy'), train_arrays_path)

    for filename in val_filenames:
        copy_image_file(images_folder, val_img_path, filename)
        shutil.copy(os.path.join(input_folder, filename + '.npy'), val_arrays_path)

    copy_folder('./carseg_data/test', './carseg_data/training_one/test')

    return "Dataset successfully prepared."

# Prepare the dataset
prepare_dataset(output_folder_combined, input_folder, training_one)


shutil.rmtree(output_folder_combined)