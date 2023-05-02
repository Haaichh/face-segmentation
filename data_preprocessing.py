import os
import numpy as np
import cv2
from utils import create_directory

FACES = 30000
FACES_PER_FOLDER = 2000
LABEL_LIST = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck', 'neck_l', 'cloth']
ROOT_DIR = os.path.dirname(os.path.abspath('data_preprocessing.py'))
MASK_DIR = os.path.join(ROOT_DIR, 'CelebAMask-HQ-mask')
ANNO_MASK_DIR = os.path.join(ROOT_DIR, 'CelebAMask-HQ/CelebAMask-HQ-mask-anno')

def create_mask():
    """Merges annotated masks into a single ground truth segmentation mask."""

    if os.path.isfile(os.path.join(MASK_DIR, '29999.png')):
        print('Masks already exist, would you like to rewrite the masks?')
        ans = input('Input [y/n]: ')
        if ans == 'n':
            return 'Did not rewrite masks'

    print('Creating masks... this may take a while')
    for image_num in range(FACES):
        folder_num = image_num // FACES_PER_FOLDER
        base_image = np.zeros((512, 512))
        for index, label in enumerate(LABEL_LIST):
            filename = os.path.join(ANNO_MASK_DIR, str(folder_num), str(image_num).rjust(5, '0') + '_' + label + '.png')
            if os.path.exists(filename):
                image = cv2.imread(filename)
                image = image[:, :, 0]
                base_image[image != 0] = index + 1
            
        cv2.imwrite((os.path.join(MASK_DIR, str(image_num) + '.png')), base_image)

    return 'Finished creating masks'

create_directory(MASK_DIR)
print(create_mask())