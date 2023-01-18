import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CelebAMask(Dataset):
    """Custom CelebAMask Dataset."""

    def __init__(self, mode):
        self.img_path = mode + '_img'
        self.mask_path = mode + '_mask'

        self.data = []
        for i in range(len(os.listdir(self.img_path))):
            self.data.append([self.img_path + '/' + str(i) + '.jpg', self.mask_path + '/' + str(i) + '.png'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img_path, mask_path = self.data[i]
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        transform = transforms.Compose([transforms.PILToTensor(), transforms.Resize(256)])
        img_tensor = transform(img).float()/255
        mask_tensor = transform(mask).float()/255

        return img_tensor, mask_tensor