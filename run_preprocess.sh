#!/bin/bash

#join all images
echo "I am preprocessing all data"
python3 data_preparing

#crop and resize all images to 256,256 - first dataset
echo "I am preparing the first dataset"
python3 dataset_one.py

#resize landscapes and prepare synthetic data
echo "I am preparing the second dataset"
python3 dataset_two.py

#echo "I am preparing the third dataset"
#python3 dataset_three.py


