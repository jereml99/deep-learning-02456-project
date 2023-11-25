import numpy as np
import os
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt

#check image size for three folders: orange_3_doors, black_5_doors, landscapes
folders_paths = ['./carseg_data/images/orange_3_doors/no_segmentation', './carseg_data/images/black_5_doors/no_segmentation', './carseg_data/images/photo/no_segmentation' ,'./carseg_data/landscapes']

for folder_path in folders_paths:
    # Set to store unique image dimensions for the current folder
    unique_dimensions = set()

    print(f"Processing folder: {folder_path}")
    
    # Iterating through files in the current folder
    for filename in os.listdir(folder_path):
        # Creating the full path to the file
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    unique_dimensions.add(img.size)
            except IOError:
                continue

    # Printing unique dimensions for the current folder
    print(f"Unique image dimensions in folder {folder_path}:")
    for dimensions in unique_dimensions:
        print(dimensions)
    print("\n")


## check .npy files
def display_class_maps(mask_folder):
    # Define distinct colors for each class
    class_colors = {
        10: 'orange',
        20: 'darkgreen',
        30: 'yellow',
        40: 'cyan',
        50: 'purple',
        60: 'lightgreen',
        70: 'blue',
        80: 'pink'
    }

    # Iterate over all .npy files in the output folder
    for file in os.listdir(mask_folder):
        if file.endswith('.npy'):
            # Load the .npy file
            data = np.load(os.path.join(mask_folder, file))
            
            # Extract the class channel
            class_channel = data[:, :, 3]

            # Create an RGBA image where each class is a different color
            rgba_image = np.zeros((class_channel.shape[0], class_channel.shape[1], 4), dtype=np.uint8)

            for class_value, color in class_colors.items():
                mask = class_channel == class_value
                rgba_image[mask] = Image.new('RGBA', (1, 1), color=color).getdata()[0]

            # Display the class map
            plt.figure(figsize=(8, 8))
            plt.imshow(rgba_image)
            plt.title(f'Class Map for {file}')
            plt.axis('off')
            plt.show()

# Specify the output folder containing .npy files
mask_folder = './carseg_data/arrays'

#display_class_maps(mask_folder)

# Conclusions:
# masks should be corrected because we see single pixels marking masks that are not part of the masks
# all .npy files are resized to the size (256, 256), we don't know what was the method of resizing, so we are not able to resize images
# we can use only images from photo's folder, because those files are in the correct size along with masks
# we should take synthetic objects from black_5_doors and orange_3_doors and put on landscapes 

# we have 139 images in 'photo' folder in size (256,256)