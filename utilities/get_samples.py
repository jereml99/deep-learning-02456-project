import os
import random
import shutil

# Paths to your directories
data_dir = 'carseg_data/arrays'
sample_dir = 'data_samples'

# Create the sample_dir if it doesn't exist
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Get all files from the data directory
all_files = os.listdir(data_dir)

random.seed(42)

# Randomly select 100 files
sample_files = random.sample(all_files, 100)

# Copy selected files to the sample directory
for file_name in sample_files:
    shutil.copyfile(os.path.join(data_dir, file_name), os.path.join(sample_dir, file_name))
