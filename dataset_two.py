
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import matplotlib.colors as mcolors


## PREPARE SYNTETIC DATA 
input_folder = './carseg_data/arrays'
landscapes_folder = './carseg_data/cropped_landscapes'
output_folder_combined = './carseg_data/combined_images'
output_folder_masks = './carseg_data/resized_masks'

# Create output folders if they don't exist
os.makedirs(output_folder_combined, exist_ok=True)
os.makedirs(output_folder_masks, exist_ok=True)

def adjust_hue(image, hue_change):
    # Convert the image to HSV
    hsv_image = mcolors.rgb_to_hsv(image/255.0)
    # Adjust the hue
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_change) % 1.0
    # Convert back to RGB
    rgb_image = mcolors.hsv_to_rgb(hsv_image)
    return (rgb_image * 255).astype(np.uint8)

def combine_with_random_background(transparent_car_image, landscapes_folder):
    # Get a list of all landscape images
    landscape_images = [f for f in os.listdir(landscapes_folder) if f.endswith(('.png', '.jpg'))]
    
    # Randomly select a landscape image
    selected_landscape = random.choice(landscape_images)
    landscape_path = os.path.join(landscapes_folder, selected_landscape)
    landscape_image = Image.open(landscape_path)

    # Resize the landscape to match the car image dimensions
    landscape_image = landscape_image.resize(transparent_car_image.size)

    # Combine the transparent car image with the landscape
    combined_image = Image.alpha_composite(landscape_image.convert('RGBA'), transparent_car_image)

    return combined_image

hue_change = 0
hue_increment = 0.05  # Adjust this value to control the hue change for each image

# Iterate through all .npy files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_folder, file_name)
        
        # Load the data from the file
        data = np.load(file_path)
        
        # Split the data into image and segmentation map
        image_data = data[..., :3]  # The first 3 channels are the image data (RGB)
        segmentation_map = data[..., 3]  # The fourth channel is the segmentation map

        # Resize the segmentation map
        resized_segmentation_map = Image.fromarray(segmentation_map).resize((256, 256), Image.NEAREST)
        resized_segmentation_map = np.array(resized_segmentation_map)

        # Save the resized segmentation map
        mask_file_path = os.path.join(output_folder_masks, file_name)
        np.save(mask_file_path, resized_segmentation_map)

        # Prepare car segments
        car_segments_mask = np.isin(resized_segmentation_map, np.arange(10, 91))

        # Isolate car segments from the image
        car_image = np.zeros_like(image_data)
        car_image[car_segments_mask] = image_data[car_segments_mask]

        # Adjust the hue of the car segments
        adjusted_car_image = adjust_hue(car_image, hue_change)
        hue_change = (hue_change + hue_increment) % 1.0

        # Convert the adjusted car segments to PIL image format with a transparent background
        transparent_background = np.zeros((adjusted_car_image.shape[0], adjusted_car_image.shape[1], 4), dtype=np.uint8)
        transparent_background[..., :3] = adjusted_car_image
        transparent_background[..., 3] = (resized_segmentation_map * (car_segments_mask * 255)).astype(np.uint8)
        car_image_transparent_pil = Image.fromarray(transparent_background)

        # Combine the car image with a random landscape background
        combined_image = combine_with_random_background(car_image_transparent_pil, landscapes_folder)

        # Save the combined image
        combined_image_filename = file_name.replace('.npy', '.png')
        combined_image.save(os.path.join(output_folder_combined, combined_image_filename))
