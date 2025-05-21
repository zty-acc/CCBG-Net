import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pylab as plt
import random
import numpy as np
import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
def Bextraction(img):
    img = img[0].numpy()
    img1 = img.astype(np.uint8)
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)
    img2 =  cv2.dilate(img1, DIAMOND_KERNEL_5).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis = 0)
    return torch.tensor(img3.copy())


class Datases_loader(Dataset):
    def __init__(self, root_images, root_masks, h, w):
        super().__init__()
        self.root_images = root_images
        self.root_masks = root_masks
        self.h = h
        self.w = w
        self.images = []
        self.labels = []

        files = sorted(os.listdir(self.root_images))
        sfiles = sorted(os.listdir(self.root_masks))
        for i in range(len(sfiles)):
            img_file = os.path.join(self.root_images, files[i])
            mask_file = os.path.join(self.root_masks, sfiles[i])
            self.images.append(img_file)
            self.labels.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image = self.images[idx]
            mask = self.labels[idx]
        else:
            image = self.images[idx]
            mask = self.labels[idx]
        image = Image.open(image)
        mask = Image.open(mask)
        tf = transforms.Compose([
            transforms.Resize((int(self.h * 1.25), int(self.w * 1.25))),
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])

        image = image.convert('RGB')
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        seed = np.random.randint(1459343089)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        img = tf(image)
        img = norm(img)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        mask = tf(mask)
        mask[mask>0] = 1.
#########################binarize segment#####################
        boundary = Bextraction(mask)

        sample = {'image': img, 'mask': mask, 'boundary': boundary}

        return sample

if __name__ == '__main__':
    imgdir = r'path01'
    labdir = r'path02'
    d = Datases_loader(imgdir, labdir, 512, 512)
    d = Datases_loader(imgdir, labdir, 512, 512)
    my_dataloader = DataLoader(d, batch_size=8, shuffle=False)
    
