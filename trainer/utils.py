#!/usr/bin/python3

import os
import json
import psutil
from PIL import Image
import cv2
import gc

import random
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
import seaborn as sns

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt  # one of the best graphics library for python
plt.style.use('ggplot')

from .configuration import SystemConfig, TrainerConfig, DataloaderConfig


class AverageMeter:
    """Class for Computing and storing the average and current value"""
    
    def __init__(self):
        """Init method of class
        """
        self.val = None
        self.avg = None
        self.reset()

    def reset(self):
        """Reset method
        """     
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        """Update method

        Args:
            val (int): input of value
            count (int, optional): number of values. Defaults to 1.
        """
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, batch_size_to_set=DataloaderConfig.batch_size):
    """Patches configs if cuda is not available

    Args:
        epoch_num_to_set (int, optional): Number of times the whole dataset will be passed through the network. Defaults to TrainerConfig.epoch_num.
        batch_size_to_set (int, optional): Amount of data to pass through the network at each forward-backward iteration. Defaults to DataloaderConfig.batch_size.

    Returns:
        dataloader_config, trainer_config: returns patched dataloader_config and trainer_config
    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 1
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, epoch_num=epoch_num_to_set, progress_bar=True)
    return dataloader_config, trainer_config


def setup_system(system_config: SystemConfig) -> None:
    """Setup System

    Args:
        system_config (SystemConfig): return configuration of system setting
    """    
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def resize(img, boxes, size, max_size=1000):
    '''Resize the input cv2 image to the given size.

    Args:
      img: (cv2) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (cv2) resized image.
      boxes: (tensor) resized boxes.
    '''
    height, width, _ = img.shape
    if isinstance(size, int):
        size_min = min(width, height)
        size_max = max(width, height)
        scale_w = scale_h = float(size) / size_min
        if scale_w * size_max > max_size:
            scale_w = scale_h = float(max_size) / size_max
        new_width = int(width * scale_w + 0.5)
        new_height = int(height * scale_h + 0.5)
    else:
        new_width, new_height = size
        scale_w = float(new_width) / width
        scale_h = float(new_height) / height

    return cv2.resize(img, (new_height, new_width)), \
           boxes * torch.Tensor([scale_w, scale_h, scale_w, scale_h])


def random_flip(img, bbox):
    '''Randomly flip the given image tensor

    Args:
        img (tensor): image tensor. shape (3, height, width)
        bbox (tensor): bounding box tensor

    Returns:
        img (tensor): randomaly fliped image tensor.
        bbox (tensor): randomaly fliped bounding box tensor
    '''
    if random.random() < 0.5:

        _, width = img.shape[-2:]
        img = img.flip(-1)
        bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        
    return img, bbox


def scale_image_with_boxes(image, boxes, scale_factor):
    """
        Scales the image and adjusts the bounding box coordinates accordingly.

    Args:
        image: Original image (numpy array).
        boxes: List of bounding boxes in the format [[x1, y1, x2, y2]].
        scale_factor: The factor by which the image will be enlarged or reduced.

    Returns:
        New image and modified bounding boxes.
    """

    # Get the dimensions of the original image
    w, h = image.size
    
    # Calculating new dimensions
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # Image scaling
    resized_image = image.resize((new_w, new_h))

    # Editing bounding box coordinates
    scaled_boxes = []
    
    for sel_list in boxes:
        scale_list = []
        scale_list = [i * scale_factor for i in sel_list]
        scaled_boxes.append(scale_list)

    return resized_image, scaled_boxes

   
def collate_fn(batch):
    """Function to convert a batch of data into a tuple of tuples.
    
        This function is typically used in the context of data processing in the PyTorch library,
        specifically when creating data loaders.

    Args:
        batch (list of lists or tuples): A list of lists or tuples, 
                                        where each element represents one item in the batch.

    Returns:
        tuple of tuples: A tuple of tuples, where each inner tuple contains elements 
                        from the same position in each element of the input batch.
    """    
    return tuple(zip(*batch))

def prediction(model, device, batch_input, max_prob=True):
    """
    get prediction for batch inputs
    """
    
    # send model to cpu/cuda according to your system configuration
    model.to(device)
    
    # it is important to do model.eval() before prediction
    model.eval()

    data = batch_input.to(device)

    output = model(data)

    # get probability score using softmax
    prob = F.softmax(output, dim=1)
    
    if max_prob:
        # get the max probability
        pred_prob = prob.data.max(dim=1)[0]
    else:
        pred_prob = prob.data
    
    # get the index of the max probability
    pred_index = prob.data.max(dim=1)[1]
    
    return pred_index.cpu().numpy(), pred_prob.cpu().numpy()


def get_target_and_prob(model, dataloader, device):
    """
    get targets and prediction probabilities
    """
    
    pred_prob = []
    targets = []
    
    for _, (data, target) in enumerate(dataloader):
        
        _, prob = prediction(model, device, data, max_prob=False)
        
        pred_prob.append(prob)
        
        target = target.numpy()
        targets.append(target)
        
    targets = np.concatenate(targets)
    targets = targets.astype(int)
    pred_prob = np.concatenate(pred_prob, axis=0)
    
    return targets, pred_prob


def get_target_and_classes_cm(model, dataloader, device):
    """
    Get true targets and predicted classes from the model.
    """
    model.eval()
    targets = []
    pred_classes = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            
            # Get model output (logits or probabilities)
            output = model(data)
            
            # Get predicted classes (use argmax to get the index of the highest probability)
            pred = torch.argmax(output, dim=1)
            
            # Append to lists
            pred_classes.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    # Convert lists to numpy arrays
    targets = np.concatenate(targets)
    pred_classes = np.concatenate(pred_classes)
    
    return targets, pred_classes

def plot_normalized_confusion_matrix(predictions, targets, class_names, norm=True):
    """
    Plots the normalized confusion matrix using seaborn and matplotlib.

    Parameters:
      - predictions: Predicted classes (cls)
      - targets: Real classes (targets)
      - class_names: List of class names
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Normalization of the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    if norm:
        # Render using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=class_names, yticklabels=class_names)

        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.show()
    else:
        # Render using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                  xticklabels=class_names, yticklabels=class_names)

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.show()


def save_model(model, device, model_dir='models', model_file_name='model.pt'):
    """Function of save model

    Args:
        model (torch.nn.Module): torch model for save
        device (torch.device): setting type of calculation device CPU/GPU. Defaults to "cuda"
        model_dir (str, optional): save directory. Defaults to 'models'.
        model_file_name (str, optional): file name of model. Defaults to 'model.pt'.
    """    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        model.to('cpu')

    # save the state_dict
    torch.save(model.state_dict(), model_path)
    
    if device == 'cuda':
        model.to('cuda')
    
    return


def load_model(model, model_dir='models', model_file_name='model.pt', weights_only=False):
    """Function for load model

    Args:
        model (torch.nn.Module): torch model for save
        model_dir (str, optional): save directory. Defaults to 'models'.
        model_file_name (str, optional): file name of model. Defaults to 'model.pt'.
        weights_only (bool, optional): value for setting load weights of model. Defaults to False.

    Returns:
        torch.nn.Module: return load model
    """    
    model_path = os.path.join(model_dir, model_file_name)

    # loading the model and getting model parameters by using load_state_dict
    model.load_state_dict(torch.load(model_path, weights_only=weights_only))
    
    return model


def memory_management(epoch):
    """Function for showing memory usage and managing resources.

    Args:
        epoch (int): Current epoch number.
    """    
    print("\n" + "=" * 40)
    print(f"[Epoch {epoch}] Resource Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"  CPU Usage     : {cpu_percent}%")
    
    # Memory usage
    memory_usage = psutil.virtual_memory()
    print(f"  RAM Usage     : {memory_usage.percent}% ({memory_usage.used / (1024 ** 3):.2f} GB / {memory_usage.total / (1024 ** 3):.2f} GB)")
    
    # Disk usage
    disk_usage = psutil.disk_usage('/')
    print(f"  Disk Usage    : {disk_usage.percent}%")
    
    # GPU statistics
    if torch.cuda.is_available():
        max_memory_allocated = torch.cuda.max_memory_allocated(device=None)
        print(f"  GPU Max Alloc : {max_memory_allocated / (1024 ** 2):.2f} MB")
        print(f"  GPU Free Mem  : {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    else:
        print("  GPU Usage     : GPU not available")
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Resource cleanup complete for epoch {epoch}")
    print("=" * 40 + "\n")
    

def id_samples(path, data_type='test'):
    """Function for selection id of samle inside dataset key is position, value is id of sample

    Args:
        path (str): path of JSON file with dataset structure
        data_type (str, optional): This is parameter of setting type of dataset: train/valid/test. Defaults to 'test'.

    Returns:
        (dict) : return dictionary of data, key is position, value is id of sample
    """    
    with open(path, 'r') as f:
        data = json.load(f)

    list_data = data['datasets'][0][data_type]
    dict_id_samples = {}

    for i, item_data in enumerate(list_data):
        dict_id_samples[i] = list(item_data.keys())[0]
        
    return dict_id_samples


def progress_bar(iterable, total, prefix=''):
    """Displays a progress bar for an iterable.

    Args:
        iterable (iterable): The iterable to iterate over.
        total (int): The total number of iterations.
        prefix (str, optional): A prefix to display before the progress bar. Defaults to ''.

    Yields:
        The next item from the iterable.
    """    
    def print_progress_bar(iteration):
        """Prints the progress bar.

        Args:
            iteration (int): The current iteration number.
        """        
        percent = ("{0:.1f}%".format(100 * (iteration / float(total))))
        filledLength = int(round(50 * iteration / float(total)))
        bar = '█' * filledLength + '-' * (50 - filledLength)
        print('\r%s |%s| %s/%s %s' % (prefix, bar, iteration, total, percent), end='\r')
        # Print New Line on Complete
        if iteration == total: 
            print()

    start = time.time()
    for i, item in enumerate(iterable):
        yield item
        print_progress_bar(i + 1)
    end = time.time()
    print("Time taken: {0}".format(round(end - start, 2)))