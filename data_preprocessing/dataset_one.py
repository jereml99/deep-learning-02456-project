import os
import shutil
import numpy as np
from PIL import Image, ImageOps
import random
import matplotlib.colors as mcolors
import cv2

# paths to folders
input_folder = './carseg_data/arrays'
all_images = './carseg_data/all_images'
landscapes_folder = './carseg_data/cropped_landscapes'
output_folder_combined = './carseg_data/combined_images'
output_folder_transformed_arrays = './carseg_data/transformed_arrays'

os.makedirs(output_folder_combined, exist_ok=True)
os.makedirs(output_folder_transformed_arrays, exist_ok=True)

# function to rotate and flip an image
def rotate_and_flip_image(image, angle, flip):
    # flip the image horizontally if flip is True
    if flip:
        image = ImageOps.mirror(image)
    # rotate the image by the given angle
    return image.rotate(angle, expand=True)

# function to adjust the hue of an image
def adjust_hue(image, hue_change):
    # convert the image to HSV format and modify the hue
    hsv_image = mcolors.rgb_to_hsv(image / 255.0)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_change) % 1.0
    # convert back to RGB format
    return (mcolors.hsv_to_rgb(hsv_image) * 255).astype(np.uint8)

# function to combine the transformed image with a random background
def combine_with_random_background(transformed_image_pil, segmentation_map_pil, landscapes_folder):
    # load a random landscape image as the background
    landscape_images = [f for f in os.listdir(landscapes_folder) if f.endswith(('.png', '.jpg'))]
    landscape_image = Image.open(os.path.join(landscapes_folder, random.choice(landscape_images)))
    landscape_image = landscape_image.resize(transformed_image_pil.size)

    # convert the landscape image to RGB if it's not
    if landscape_image.mode != 'RGB':
        landscape_image = landscape_image.convert('RGB')

    # convert transformed image and segmentation map to numpy arrays
    transformed_image_array = np.array(transformed_image_pil)
    segmentation_map_array = np.array(segmentation_map_pil)
    landscape_image_array = np.array(landscape_image)

    # Replace pixels in the landscape with pixels from the transformed image
    landscape_image_array[segmentation_map_array != 0] = transformed_image_array[segmentation_map_array != 0]

    # convert the modified landscape array back to PIL image
    combined_image = Image.fromarray(landscape_image_array)

    return combined_image


# setting up hue changes for color variation
hue_changes = np.linspace(0, 1, 20, endpoint=False)  # 20 different colors
hue_index = 0

# process each .npy file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_folder, file_name)
        data = np.load(file_path)

        # split the data into RGB image data and segmentation map
        image_data, segmentation_map = data[..., :3], data[..., 3]

        # apply hue adjustment
        hue_change = hue_changes[hue_index]
        image_data = adjust_hue(image_data, hue_change)

        # choose random rotation angle and flip
        rotation_angle = random.choice([0, 20, 30 ,35 ,40 , 50, 60, 70, 80, 90, 100, 110, 120])
        flip = random.choice([True, False])

        # convert numpy arrays to PIL images and apply transformations
        image_data_pil = Image.fromarray(image_data)
        segmentation_map_pil = Image.fromarray(segmentation_map).convert('L')

        transformed_image_data_pil = rotate_and_flip_image(image_data_pil, rotation_angle, flip)
        transformed_segmentation_map_pil = rotate_and_flip_image(segmentation_map_pil, rotation_angle, flip)

        # combine the transformed image with a random background
        combined_image = combine_with_random_background(transformed_image_data_pil, transformed_segmentation_map_pil, landscapes_folder)
        combined_image.save(os.path.join(output_folder_combined, file_name.replace('.npy', '.png')))

        # re-combine the transformed RGB data and segmentation map into a single array
        transformed_image_data = np.array(transformed_image_data_pil)
        transformed_segmentation_map = np.array(transformed_segmentation_map_pil)
        transformed_data_with_segmentation = np.dstack((transformed_image_data, transformed_segmentation_map))

        # save the combined data as a .npy file
        np.save(os.path.join(output_folder_transformed_arrays, file_name), transformed_data_with_segmentation)

        # update hue change index for the next image
        hue_index = (hue_index + 1) % len(hue_changes)





### PREPARE MORE AUGMENTED 'REAL IMAGES' 
        
def augment_real_data(input_folder_images, input_folder_arrays, input_folder_backgrounds, number_of_new_images, output_folder_images, output_folder_arrays):

    images = [f for f in os.listdir(input_folder_images) if f.endswith('.jpg')]
    backgrounds = os.listdir(input_folder_backgrounds)

    for image_name, background_name in zip(images, backgrounds):
        array_name = image_name.replace('.jpg', '.npy')
        image_path = os.path.join(input_folder_images, image_name)
        array_path = os.path.join(input_folder_arrays, array_name)
        background_path = os.path.join(input_folder_backgrounds, background_name)

        if os.path.exists(array_path) and os.path.exists(background_path):
            image = cv2.imread(image_path)
            array = np.load(array_path)
            background = cv2.imread(background_path)

            for i in range(number_of_new_images):
                angle = random.randint(0, 360)
                flip = random.choice([True, False])

                augmented_image, augmented_array = augment_segments(image, array, background, angle, flip)

                save_path_image = os.path.join(output_folder_images, f'{os.path.splitext(image_name)[0]}_aug_{i}.jpg')
                save_path_array = os.path.join(output_folder_arrays, f'{os.path.splitext(array_name)[0]}_aug_{i}.npy')

                cv2.imwrite(save_path_image, augmented_image)
                np.save(save_path_array, augmented_array)

def augment_segments(image, array, background, angle, flip):
    augmented_image = background.copy()
    augmented_array = np.zeros_like(array)  

    for cls in np.unique(array[:, :, 3]): 
        if cls != 0:
            class_mask = (array[:, :, 3] == cls).astype(np.uint8) * 255

            segmented_image = cv2.bitwise_and(image, image, mask=class_mask)

            rotated_segment = rotate_image(segmented_image, angle)
            rotated_mask = rotate_image(class_mask, angle)

            if flip:
                rotated_segment = cv2.flip(rotated_segment, 1)
                rotated_mask = cv2.flip(rotated_mask, 1)

            augmented_image = apply_segment(augmented_image, rotated_segment, rotated_mask)
            augmented_array[:, :, 3] = np.where(rotated_mask > 0, cls, augmented_array[:, :, 3])

    return augmented_image, augmented_array

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

def apply_segment(base_image, segment_image, mask):
    foreground = cv2.bitwise_and(segment_image, segment_image, mask=mask)
    background = cv2.bitwise_and(base_image, base_image, mask=cv2.bitwise_not(mask))
    return cv2.add(background, foreground)


augment_real_data(all_images, input_folder, landscapes_folder , 20, output_folder_combined, output_folder_transformed_arrays)



### PREPARE THE DATASET
training_one = './carseg_data/training_one'

def copy_folder(src, dst):
    if os.path.exists(dst):
        print("The output folder exists.")
    else:
        # Copy the folder with all its content
        shutil.copytree(src, dst)
        print(f"The folder '{src}' was copied to location '{dst}'")

def prepare_dataset(images_folder, masks_folder, output_folder):
    # paths for the output subfolders
    train_img_path = os.path.join(output_folder, 'train', 'img')
    train_arrays_path = os.path.join(output_folder, 'train', 'arrays')
    val_img_path = os.path.join(output_folder, 'val', 'img')
    val_arrays_path = os.path.join(output_folder, 'val', 'arrays')

    # create necessary directories
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_arrays_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(val_arrays_path, exist_ok=True)

    # get all image and mask filenames
    images = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg'))]
    masks = [f for f in os.listdir(masks_folder) if f.endswith('.npy')]

    # filter out the common filenames
    common_filenames = set([os.path.splitext(f)[0] for f in images]) & set([os.path.splitext(f)[0] for f in masks])

    # function to check and copy the corresponding image file
    def copy_image_file(src_folder, dest_folder, filename):
        for ext in ['.jpg', '.png']:
            if os.path.exists(os.path.join(src_folder, filename + ext)):
                shutil.copy(os.path.join(src_folder, filename + ext), dest_folder)
                break

    # split data into training and validation sets (5% for validation)
    val_filenames = set(np.random.choice(list(common_filenames), size=int(len(common_filenames) * 0.05), replace=False))
    train_filenames = common_filenames - val_filenames

    # copy files to their respective directories
    for filename in train_filenames:
        copy_image_file(images_folder, train_img_path, filename)
        shutil.copy(os.path.join(masks_folder, filename + '.npy'), train_arrays_path)

    for filename in val_filenames:
        copy_image_file(images_folder, val_img_path, filename)
        shutil.copy(os.path.join(masks_folder, filename + '.npy'), val_arrays_path)

    copy_folder('./carseg_data/test', './carseg_data/training_one/test')

    print("The dataset one successfully prepared.")

# prepare the dataset
prepare_dataset(output_folder_combined, output_folder_transformed_arrays, training_one)


shutil.rmtree(output_folder_combined)
shutil.rmtree(output_folder_transformed_arrays)
