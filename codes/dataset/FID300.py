import numpy as np
import cv2
import os

class FID300(object):

    def __init__(self, input_dir, h=192, w=384, print_dir='print', mask_dir='masks'):
        self.input_dir = input_dir
        self.height = h
        self.width = w

        label_name = os.path.join(input_dir, 'labels.npy')

        self.labels = np.load(label_name, allow_pickle=True).item()
        for label in sorted(self.labels.keys()):
            self.labels[label] = set(self.labels[label].split('_'))

        reference_map_name = os.path.join(self.input_dir, 'labels_crime_scene_to_reference_map.npy')
        crime_print_to_reference_map = np.load(reference_map_name, allow_pickle=True)
        from collections import defaultdict
        labels = defaultdict(list)
        labels.update({crime_print_to_reference_map[i, 0]: self.labels[crime_print_to_reference_map[i, 1].zfill(6)] for i in range(crime_print_to_reference_map.shape[0])})
        self.labels = labels

        mask_dir = os.path.join(self.input_dir, 'mask')
        print_dir = os.path.join(self.input_dir, print_dir)
        mask_dir = os.path.join(self.input_dir, mask_dir)

        self.print_file_names = np.array(sorted([f for f in os.listdir(print_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]))

        self.print_files = np.empty(len(self.print_file_names), dtype=object)
        self.mask_files = np.empty(len(self.print_file_names), dtype=object)

        for i, file in enumerate(self.print_file_names):
            self.print_files[i] = os.path.join(print_dir, file)
            self.mask_files[i] = os.path.join(mask_dir, file)

        print(f"Total shoes for FID {print_dir.rsplit('/', 1)[1]}: {len(self.print_files)}")

    def get_all_matches(self):
        return self.labels

    def __getitem__(self, index):
        image = cv2.imread(self.print_files[index]) #[:,:,0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if os.path.exists(self.mask_files[index]):
            mask = cv2.imread(self.mask_files[index])[:,:,0]
        else:
            mask = np.ones(image.shape)*255
        dilation_size = 8
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)
        image = (image[np.newaxis, ...]/255.0).astype(np.float32)
        mask = mask[np.newaxis, ...] > 255/2
        return image, mask, self.print_file_names[index]

    def __len__(self):
        return len(self.print_files)