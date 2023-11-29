#!/bin/bash

#join all images
echo "I am preprocessing all data with cropped landscapes, correct arrays"
python3 ./deep-learning-02456-project/data_preparing.py

#synthetic dataset with 20 colours of synthetic cars
echo "I am preparing the first dataset"
python3 ./deep-learning-02456-project/dataset_one.py

#synthetic dataset with 10 colours of synthetic cars
echo "I am preparing the second dataset"
python3 ./deep-learning-02456-project/dataset_two.py



