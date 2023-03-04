# Load save model as h5 file

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from os.path import dirname, abspath, join
from typing import Optional

def load_model_h5(model_name:str) -> Optional[keras.Model]:
    try:
        model = load_model(model_name)
    except OSError:
        print('Model not found with name: ', model_name)
        return None
    return model


if __name__ == '__main__':

    model_name = 'mlp_base_model_raw_cv0.h5'

    # Get path
    parent_folder = dirname(dirname(abspath(__file__)))
    print(parent_folder)
    model_path = join(parent_folder, 'model/single_run', model_name)
    print(model_path)

    # Load model
    model = load_model_h5(model_path)

    # Print model summary
    try:
        print(model.summary())
        print(model.score())
    except AttributeError:
        print('Model not loaded properly')

