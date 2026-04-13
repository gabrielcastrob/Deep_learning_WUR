



# Function to load packages 
def load_packages():
    import random, shutil, zipfile
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import torchvision.models as tvm
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    import torchmetrics

    L.seed_everything(42, workers=True)
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator:", DEVICE)
    if DEVICE == "gpu":
        print("GPU:", torch.cuda.get_device_name(0))

    return (random, shutil, zipfile, Path, np, pd, torch, nn, Dataset, DataLoader, 
            transforms, tvm, Image, train_test_split, accuracy_score, f1_score, confusion_matrix, plt, L, 
            ModelCheckpoint, CSVLogger, torchmetrics)


def download_ucm_dataset():
    "Download the UCMerced Land Use dataset if not already present. "
    "The dataset will be saved in the 'ucmdata' directory. "
    
    import os
    import zipfile
    import subprocess
    import shutil
    if not os.path.exists('ucmdata'):
        subprocess.run(['git', 'clone', 'https://git.wur.nl/lobry001/ucmdata.git'])
        os.chdir('ucmdata')

        with zipfile.ZipFile('UCMerced_LandUse.zip', 'r') as zip_ref:
            zip_ref.extractall('UCMImages')

        shutil.move('UCMImages/UCMerced_LandUse/Images', '.')
        shutil.rmtree('UCMImages')
        os.remove('README.md')
        os.remove('UCMerced_LandUse.zip')
        print(os.listdir('.'))
        UCM_images_path = "Images/"
        Multilabels_path = "LandUse_Multilabeled.txt"
    return UCM_images_path, Multilabels_path
        
