# Face Segmentation Using U-Net Based Architectures
--------------------------
## Dataset:
- CelebAMask-HQ: https://github.com/switchablenorms/CelebAMask-HQ
- CelebA-HQ-to-CelebA-napping.txt: Found under Train/Val/Test Partitions at https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Unzip dataset to /face-segmentation/

## Dependencies:
- Python v.3.8.10: https://www.python.org/downloads/
- PyTorch: https://pytorch.org/get-started/locally/
- Matplotlib: https://matplotlib.org/stable/users/installing/index.html
- scikit-learn: https://scikit-learn.org/stable/install.html
- pandas: https://pandas.pydata.org/docs/getting_started/install.html
- OpenCV2: https://opencv.org/releases/

## Instructions:
- Run data_preprocessing.py - This creates the ground truth segmentation masks and stores them in /face-segmentation/CelebAMask-HQ-mask/
- Run data_split.py - This splits the data into the training, testing, and validation sets. /test_img/, /test_mask/, /train_img/, /train_mask/, /val_img/, and /val_mask/ are all stored in /face-segmentation/
- Run train_test.py - This runs the training and testing loops for the specified model. Produces a .pt file containing the model state after taining. Produces five .txt files containing accuracy, loss, and F1-score statistics.
- Run validation.py - This loads the .pt model file and calculates the F1-score, loss, and accuracy on the validation set.
