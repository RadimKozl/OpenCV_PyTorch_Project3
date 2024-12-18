#!/usr/bin/python3

"""Module custom dataloader

This module store class for create PyTorch Dataloader from files
"""

# Import libraries
import os
import json
import sys
import h5py
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .utils import random_flip, scale_image_with_boxes


class ListDataset(Dataset):
    """Class for load dataset
        
        This class load dataset with specific structure of directories 
        and labels *.csv files. 
        
        Origin dateset was: https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection
    
    Args:
        Dataset (class): PyTorch Dataset class
    """    
    def __init__(self, csv_path, train=False, transform=None):
        '''Init method of class
        
        Args:
          csv_path: (str) ditectory to images.
          train: (boolean) True if train else False.
          transform: ([transforms]) image transforms.
        '''
        
        self.transform = transform
        self.train = train

        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(csv_path) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1 + 5 * i]
                ymin = splited[2 + 5 * i]
                xmax = splited[3 + 5 * i]
                ymax = splited[4 + 5 * i]
                class_label = splited[5 + 5 * i]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(class_label))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image method

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          targets: (dict) location and bbox targets.
        '''
        # Load image and boxes.
        img = Image.open(self.fnames[idx]).convert("RGB")
        # transform PIL to tensor
        img = F.to_tensor(img)
        
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]        
            
        # Data augmentation.
        if self.transform is not None:
            img = self.transform(img)
            
        if self.train:
            img, boxes = random_flip(img, boxes)
            
        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target

    def __len__(self):
        """Method of len() function

        Returns:
            int: number of samples
        """        
        return self.num_samples
    
class JsonClassificationDataset(Dataset):
    """Class for create Dataset for PyTorch from JSON file data

    Args:
        Dataset (class): PyTorch Dataset class
    """    
    def __init__(self, json_file, type_data='train', dataset_number=0, image_shape=None, transform=None, ):
        """Init method of class

        Args:
            json_file (str): path of JSON file with data of datasets
            type_data (str, optional): Setting of load type of dataset - 'train'/'valid'/'test' data. Defaults to 'train'.
            dataset_number (int, optional): This is number of creted variant of datasets. Defaults to 0.
            image_shape (int/tuple, optional): value of weight & height of resized image. Defaults to None.
            transform (torchvision.transforms.Compose, optional): list of transformation of PyTorch. Defaults to None.

        """
        super().__init__()
        self.json_file = json_file
        with open(self.json_file, 'r', encoding="utf-8") as f:
            self.config_datatasets = json.load(f)

        self.dataset_number = int(dataset_number)
        self.type_data = type_data
        self.base_setting = list(self.config_datatasets['datasets'][0].keys())

        # set image_resize attribute
        if image_shape is not None:
            if isinstance(image_shape, int):
                self.image_shape = (image_shape, image_shape)

            elif isinstance(image_shape, tuple) or isinstance(image_shape, list):
                assert len(image_shape) == 1 or len(image_shape) == 2, 'Invalid image_shape tuple size'
                if len(image_shape) == 1:
                    self.image_shape = (image_shape[0], image_shape[0])
                else:
                    self.image_shape = image_shape
            else:
                raise NotImplementedError

        else:
            self.image_shape = image_shape

        # set transform attribute
        self.transform = transform

        self.num_classes = self.config_datatasets['datasets'][self.dataset_number]['class_number']

        # initialize the data dictionary
        self.data_dict = {
            'image_path': [],
            'label': []
        }

        self._load_dataset()

    def _load_train_dataset(self):
        """Funcion for load train dataset
        """        
        number_records = len(self.config_datatasets['datasets'][self.dataset_number]['train'])

        for num in range(number_records):
            value_data = list(self.config_datatasets['datasets'][self.dataset_number]['train'][num].values())
            self.data_dict['image_path'].append(value_data[0]['path'])
            self.data_dict['label'].append(value_data[0]['clidx'])

    def _load_valid_dataset(self):
        """Function for load valid dataset
        """        
        number_records = len(self.config_datatasets['datasets'][self.dataset_number]['valid'])

        for num in range(number_records):
            value_data = list(self.config_datatasets['datasets'][self.dataset_number]['valid'][num].values())
            self.data_dict['image_path'].append(value_data[0]['path'])
            self.data_dict['label'].append(value_data[0]['clidx'])

    def _load_test_dataset(self):
        """Function for load test dataset
        """        
        number_records = len(self.config_datatasets['datasets'][self.dataset_number]['test'])

        for num in range(number_records):
            value_data = list(self.config_datatasets['datasets'][self.dataset_number]['test'][num].values())
            self.data_dict['image_path'].append(value_data[0]['path'])
            self.data_dict['label'].append(value_data[0]['clidx'])

    def _load_dataset(self):
        """Internal Method for load selected dataset

        Returns:
            str: if is wrong selection parameter, it was returned warring message.
        """        
        if self.type_data == 'train':
            if 'train' in self.base_setting:
                self._load_train_dataset()
            else:
                print('Json file does not contain dataset train')

        elif self.type_data == 'valid':
            if 'valid' in self.base_setting:
                self._load_valid_dataset()
            else:
                print('Json file does not contain dataset valid')

        elif self.type_data == 'test':
            if 'test' in self.base_setting:
                self._load_test_dataset()
            else:
                print('Json file does not contain dataset test')
        else:
            return 'False settings of type_data parameter.'

    def __len__(self):
        """Method of return length of the dataset
        
        Returns:
            str: if is wrong selection parameter, it was returned warring message.
        """
        if self.type_data == 'train':
            return len(self.config_datatasets['datasets'][self.dataset_number]['train'])
        elif self.type_data == 'valid':
            return len(self.config_datatasets['datasets'][self.dataset_number]['valid'])
        elif self.type_data == 'test':
            return len(self.config_datatasets['datasets'][self.dataset_number]['test'])
        else:
            return 'False settings of type_data parameter.'

    def __getitem__(self, idx):
        """Method for given index, return images with resize and preprocessing.

        Args:
            idx (int): number of index of image

        Returns:
            np.array, int: return image as np.array and number of class as int
        """        
        image = Image.open(self.data_dict['image_path'][idx]).convert("RGB")

        if self.image_shape is not None:
            image = F.resize(image, self.image_shape)

        if self.transform is not None:
            image = self.transform(image)

        target = self.data_dict['label'][idx]

        return image, target


    def common_name(self, label):
        """
        Method of class label to common name mapping
        """
        list_labels = list(self.config_datatasets['datasets'][0]['names_class'])
        return list_labels[label]
    
    def number_of_class(self):
        """Method for return number of class of datasets

        Returns:
            int: number of class of dataset
        """        
        return self.num_classes
    
    def names_of_class(self):
        """Method return names of classes

        Returns:
            list(str): list of names of class
        """        
        return list(self.config_datatasets['datasets'][0]['names_class'])
    
    def calculate_mean_std_manual(self):
        """Method for manually calculating mean and standard deviation of the dataset.

        Returns:
            tuple: mean and std for each channel (R, G, B)
        """
        mean = np.zeros(3)
        mean_sqrd = np.zeros(3)
        n_pixels = 0

        for img_path in self.data_dict['image_path']:
            # Load image
            image = Image.open(img_path).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]

            # Add to the total pixel count
            n_pixels += image.shape[0] * image.shape[1]

            # Calculate per-channel mean and squared mean
            mean += np.mean(image, axis=(0, 1))
            mean_sqrd += np.mean(image ** 2, axis=(0, 1))

        # Calculate final mean
        mean /= len(self.data_dict['image_path'])

        # Calculate variance and std (per-channel)
        variance = (mean_sqrd / len(self.data_dict['image_path'])) - (mean ** 2)
        std = np.sqrt(variance)

        print(f"Mean: {mean}, Std: {std}")
        return mean, std


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, dataset_type, transform=None, train=False, width_image=800):
        self.hdf5_file = hdf5_file
        self.dataset_type = dataset_type
        self.transform = transform
        self.train = train
        self.width_image = width_image

        # Load minimal information to initialize dataset
        with h5py.File(self.hdf5_file, 'r') as database:
            self.list_files = list(database[self.dataset_type].keys())
            self.database_attribute = list(database.attrs.keys())
            self.class_number = int(database.attrs[self.database_attribute[0]])
            self.names_class = list(database.attrs[self.database_attribute[1]])

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        image_name = self.list_files[idx]

        with h5py.File(self.hdf5_file, 'r') as database:
            dataset = database[self.dataset_type]
            image_link = dataset[image_name]['image_link'][:].tolist()[0].decode('UTF-8')

            boxes_values = dataset[image_name].attrs['boxes'].tolist()
            image_labels = dataset[image_name].attrs['labels'].tolist()

        # Load image
        if not os.path.exists(image_link):
            raise FileNotFoundError(f"Image path {image_link} does not exist.")
        
        img = Image.open(image_link).convert("RGB")
        w_img = img.width
        
        if self.train:
            scale_factor = round(self.width_image / w_img, 2)
            img, boxes_values = scale_image_with_boxes(img, boxes_values, scale_factor)
        
        img = F.to_tensor(img)

        # Prepare bounding boxes and labels
        boxes = torch.tensor(boxes_values, dtype=torch.float32)
        labels = torch.tensor(image_labels, dtype=torch.int64)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.train:
            img, boxes = random_flip(img, boxes)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target

    def calculate_mean_std_manual(self):
        mean = np.zeros(3)
        mean_sqrd = np.zeros(3)
        n_pixels = 0

        for name_file in self.list_files:
            try:
                with h5py.File(self.hdf5_file, 'r') as database:
                    dataset = database[self.dataset_type]
                    image_link = dataset[name_file]['image_link'][:].tolist()[0].decode('UTF-8')

                if not os.path.exists(image_link):
                    print(f"Skipping image {image_link} as it does not exist.")
                    continue

                img = Image.open(image_link).convert("RGB")
                img_np = np.array(img, dtype=np.float32) / 255.0

                mean += img_np.mean(axis=(0, 1))
                mean_sqrd += (img_np ** 2).mean(axis=(0, 1))
                n_pixels += 1

            except Exception as e:
                print(f"Skipping image {name_file} due to error: {e}")

        mean /= n_pixels
        mean_sqrd /= n_pixels
        std = np.sqrt(mean_sqrd - mean ** 2)

        return mean, std
