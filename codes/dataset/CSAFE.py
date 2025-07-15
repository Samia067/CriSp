import numpy as np
import cv2
import os


class Category:
    def __init__(self, name, path, print_dir='print', mask_dir='mask'):
        self.name = name
        self.path = path
        self.print_dir = os.path.join(self.path, print_dir)
        self.mask_dir = os.path.join(self.path, mask_dir)

        self.file_names = np.array(sorted([f for f in os.listdir(self.print_dir)
                                               if (f.lower().endswith('.jpg') or f.lower().endswith('.png'))]))
        self.print_files = np.empty(len(self.file_names), dtype=object)
        self.mask_files = np.empty(len(self.file_names), dtype=object)
        for i, file in enumerate(self.file_names):
            self.print_files[i] = os.path.join(self.print_dir, file)
            self.mask_files[i] = os.path.join(self.mask_dir, file)
        print("Total shoes in", self.name, ": ", len(self.print_files))

        self.names = []
        for name in self.file_names:
            parts = name[:-4].split('_')
            self.names.append(parts[1])


class CSAFE():
    category_names = ['dust', 'blood']

    def __init__(self, input_dir, h=192, w=384,
                 print_dir='print', mask_dir='mask'):

        self.input_dir = input_dir
        self.height = h
        self.width = w

        self.print_dir = print_dir
        self.mask_dir = mask_dir
        self.make_models = self.load_make_models()

        self.categories = {category:self.initialize_category(category) for category in self.category_names}

        self.targets = self.load_targets(self.get_all_make_models())
        self.targets = {mm: self.targets[self.make_models[mm]] for mm in self.make_models}
        self.cur_category = None

    def load_make_models(self):
        make_models = np.load(os.path.join(self.input_dir, 'make_models.npy'))
        make_models = {make_models[i, 0]:make_models[i, 1].lower().replace(" ", '_') for i in range(len(make_models))}
        return make_models

    def load_targets(self, make_models):
        targets = {}
        for make_model in make_models:
            make_model = make_model.lower().replace(' ', '_')
            targets[make_model] = set(
                np.load(os.path.join(self.input_dir, 'make_model_mapping', make_model + '_similar.npy')))
        return targets

    def get_all_make_models(self):
        return np.unique(list(self.make_models.values()))

    def initialize_category(self, category_name):
        category = Category(category_name, os.path.join(self.input_dir, category_name.split('_')[0]),
                            print_dir=self.print_dir, mask_dir=self.mask_dir)

        # sanity check
        for name in category.names:
            assert (name in self.make_models)

        return category
    def set_category(self, category):
        assert (category in self.category_names)
        self.cur_category = category

    def get_categories(self):
        return self.category_names

    def get_all_matches(self):
        return self.targets

    def __getitem__(self, index):
        category = self.categories[self.cur_category]

        image = cv2.imread(category.print_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if os.path.exists(category.mask_files[index]):
            mask = cv2.imread(category.mask_files[index])[:,:,0]
        else:
            mask = np.ones(image.shape)

        # Taking a matrix of size 5 as the kernel
        dilation_size = 8
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=5)
        image = (image[np.newaxis, ...]/255.0).astype(np.float32)
        mask = mask[np.newaxis, ...] > 255/2

        return image, mask, category.names[index]

    def __len__(self):

        for category in self.get_categories():
            if self.cur_category == category:
                return len(self.categories[category].print_files)

        return 0