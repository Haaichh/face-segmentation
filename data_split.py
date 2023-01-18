import os
import pandas as pd
from shutil import copyfile
from utils import create_directory

ROOT_DIR = os.path.dirname(os.path.abspath('data_split.py'))
MASK_DIR = os.path.join(ROOT_DIR, 'CelebAMask-HQ-mask')
IMG_DIR = os.path.join(ROOT_DIR, 'CelebAMask-HQ/CelebA-HQ-img')
MAPPING_TXT = os.path.join(ROOT_DIR, 'CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt')

def train_test_val():
    # Splits data into train, test, and validation sets

    # Create directories for the split data
    train_mask = 'train_mask'
    train_img = 'train_img'
    test_mask = 'test_mask'
    test_img = 'test_img'
    val_mask = 'val_mask'
    val_img = 'val_img'
    folders = [train_mask, train_img, test_mask, test_img, val_mask, val_img]
    for folder in folders:
        create_directory(os.path.join(ROOT_DIR, folder))

    if os.path.isfile(os.path.join(train_img, '24182.jpg')):
        print('Data split has already been completed, would you like to redo the split?')
        ans = input('Input [y/n]: ')
        if ans == 'n':
            return 0

    train, test, val = 0, 0, 0
    image_list = pd.read_csv(MAPPING_TXT, delim_whitespace=True, header=None)

    for i, x in enumerate(image_list.loc[1:, 1]):
        x = int(x)
        if x >= 162771 and x < 182638:
            copyfile(os.path.join(MASK_DIR, str(i)+'.png'), os.path.join(val_mask, str(val)+'.png'))
            copyfile(os.path.join(IMG_DIR, str(i)+'.jpg'), os.path.join(val_img, str(val)+'.jpg'))        
            val += 1
        elif x >= 182638:
            copyfile(os.path.join(MASK_DIR, str(i)+'.png'), os.path.join(test_mask, str(test)+'.png'))
            copyfile(os.path.join(IMG_DIR, str(i)+'.jpg'), os.path.join(test_img, str(test)+'.jpg'))
            test += 1 
        else:
            copyfile(os.path.join(MASK_DIR, str(i)+'.png'), os.path.join(train_mask, str(train)+'.png'))
            copyfile(os.path.join(IMG_DIR, str(i)+'.jpg'), os.path.join(train_img, str(train)+'.jpg'))
            train += 1  

    return train+test+val

print('Data split complete on', train_test_val(), 'images')