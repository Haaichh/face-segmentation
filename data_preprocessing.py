import os

# Create directory for masks
parentDir = 'git/face-segmentation'
masksDir = 'CelebAMask-HQ-masks'
path = os.path.join(os.getcwd(), parentDir, masksDir)

try:
    os.mkdir(path)
    print('Directory ' + masksDir + ' created')
except FileExistsError:
    print('Directory ' + masksDir + ' already exists')