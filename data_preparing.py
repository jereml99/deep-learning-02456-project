import os
import shutil
import numpy as np
from PIL import Image

## Create one dataset with all images without segmentation

main_directory = './carseg_data/images'
result_directory = './carseg_data/all_images'

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)

    if os.path.isdir(folder_path):
        no_seg_path = os.path.join(folder_path, 'no_segmentation')
        
        if os.path.exists(no_seg_path):
            
            for file in os.listdir(no_seg_path):
                file_path = os.path.join(no_seg_path, file)
                new_file_name = folder + '_' + file
                new_file_path = os.path.join(result_directory, new_file_name)
                shutil.copy(file_path, new_file_path)

print("I have prepared folder (with) all_images.")



## PREPARE LANDSCAPES
def crop_images_in_folder(source_folder, output_folder, target_width, target_height):
   
    os.makedirs(output_folder, exist_ok=True)

    # Process all images and crop them
    for image_file in os.listdir(source_folder):
        if image_file.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(source_folder, image_file)
            image = Image.open(image_path)

            # Check if the image is large enough to crop
            if image.size[0] >= target_width and image.size[1] >= target_height:
                # Crop the image
                cropped_image = crop_image_center(image, target_width, target_height)

                # Save the cropped image
                cropped_image_file = image_file
                cropped_image.save(os.path.join(output_folder, cropped_image_file))
            else:
                continue

def crop_image_center(image, target_width, target_height):
    """
    Crops an image to the target width and height from the center.
    """
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2

    # Crop the center of the image
    return image.crop((left, top, right, bottom))

folder = './carseg_data/landscapes'
output = './carseg_data/cropped_landscapes'
target_width = 256
target_height = 256

# Crop images in both folders
crop_images_in_folder(folder, output, target_width, target_height)
