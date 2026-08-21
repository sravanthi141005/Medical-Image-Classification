# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

print(f"TF: {tf.__version__}, Keras: {keras.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

with open('../data/data_config.pkl', 'rb') as f:
    config = pickle.load(f)

CLASSES    = config['classes']
IMAGE_SIZE = config['image_size']
test_df    = config['test_df']
SEG_DIR    = '../data/segmented_images'

print(f"Classes: {len(CLASSES)} → {CLASSES}")
print(f"Test set: {len(test_df):,}")

# %%
