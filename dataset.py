import os
import torch
import cv2
from torch.utils.data import Dataset
import config


class MapDataset(Dataset):
    """
    Preparing and Augmenting Dataset
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_item = os.listdir(self.root_dir)
        print(self.list_item)

    def __len__(self):
        return len(self.list_item)

    def __getitem__(self, index):

        img_files = self.list_item[index]
        path = os.path.join(self.root_dir, img_files)
        image = cv2.imread(path)
        #print(image.shape)
        input_img = image[:, 256:, :]
        target_img = image[:, :256, :]

        augmentations = config.both_transform(image=input_img, image0=target_img)
        input_img, target_img = augmentations['image'], augmentations['image0']

        input_img = config.transform_only_input(image=input_img)["image"]
        target_img = config.transform_only_target(image=target_img)["image"]

        return input_img, target_img