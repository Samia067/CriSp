import numpy as np
import cv2
import os
from codes.utils import get_invalid_tensor
from codes.dataset.utils import read_image, image_to_channels

class RefDb(object):

    def __init__(self, input_dir, opt, dataset_size=None, albedo_dir='albedo', depth_dir='depth',
                 image_dir='image', get_image=False, get_depth=True, get_print=False, get_mask=True):
        self.input_dir = input_dir
        self.opt = opt

        image_dir = os.path.join(self.input_dir, image_dir)
        self.get_image = os.path.exists(image_dir)*get_image
        mask_dir = os.path.join(self.input_dir, 'mask')
        print_dir = os.path.join(self.input_dir, 'print')
        self.get_print = os.path.exists(print_dir) * get_print
        albedo_dir = os.path.join(self.input_dir, albedo_dir)
        self.get_albedo = os.path.exists(albedo_dir)
        depth_dir = os.path.join(self.input_dir, depth_dir)
        self.get_depth = os.path.exists(depth_dir) * get_depth
        self.get_mask = get_mask * os.path.exists(mask_dir)
        dir = image_dir if self.get_image else print_dir

        self.image_file_names = np.array(sorted([f for f in os.listdir(dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]))

        self.image_files = np.empty(len(self.image_file_names), dtype=object) # []
        self.mask_files = np.empty(len(self.image_file_names), dtype=object)
        self.print_files = np.empty(len(self.image_file_names), dtype=object)
        if self.get_albedo:
            self.albedo_files = np.empty(len(self.image_file_names), dtype=object)
        if self.get_depth:
            self.depth_files = np.empty(len(self.image_file_names), dtype=object)

        for i, file in enumerate(self.image_file_names):
            self.image_files[i] = os.path.join(image_dir, file)
            self.mask_files[i] = os.path.join(mask_dir, file)
            self.print_files[i] = os.path.join(print_dir, file)
            if self.get_albedo:
                self.albedo_files[i] = os.path.join(albedo_dir, file)
            if self.get_depth:
                self.depth_files[i] = os.path.join(depth_dir, file)

        print("Total shoes: ", len(self.image_files))
        if dataset_size is None:
            self.dataset_size = len(self.image_files)
        else:
            self.dataset_size = dataset_size

        labels = [n[:6] for n in self.image_file_names]
        labels, counts = np.unique(labels, return_counts=True)
        self.labels_map = {label: index for index, label in enumerate(labels)}
        self.labels_freq = {self.labels_map[label]: count for label, count in zip(labels, counts)}

    def __getitem__(self, index):
        index = index % len(self.image_files)
        if self.get_mask:
            mask = read_image(self.mask_files[index], is_mask=True)
            mask = mask.astype(np.float)
            mask = np.round(mask).astype(bool)

        if self.get_image:
            image = read_image(self.image_files[index]) if self.get_image else get_invalid_tensor(tensor=False)
            if self.get_mask:
                mask3d_inverted = ~mask.repeat(3, axis=2)
                image[mask3d_inverted] = 1
            image = image_to_channels(image)
        else:
            image = get_invalid_tensor(tensor=False)

        if self.get_mask:
            mask = image_to_channels(mask)
            mask = mask[0:1, ...].astype(np.bool)
        else:
            mask = get_invalid_tensor(tensor=False)

        if self.get_print:
            print_ = read_image(self.print_files[index])
            print_ = image_to_channels(print_)
            print_ = print_[0:1, ...]
            print_ += 0.3*(np.random.random(print_.shape)-0.5)
        else:
            print_ = get_invalid_tensor(tensor=False)

        if self.get_albedo:
            albedo = read_image(self.albedo_files[index])
            albedo = image_to_channels(albedo)
        else:
            albedo = get_invalid_tensor(tensor=False)

        if self.get_depth:
            depth = read_image(self.depth_files[index], gray=True)
            depth = image_to_channels(depth)
        else:
            depth = get_invalid_tensor(tensor=False)

        return image, mask, print_, albedo, depth, self.image_file_names[index]

    def __len__(self):
        return self.dataset_size