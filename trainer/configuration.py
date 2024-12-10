#!/usr/bin/python3
"""Configurations module
"""

# Imports libraries
import os
import yaml

from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision.transforms import ToTensor


@dataclass
class SystemConfig:
    """Class of System Configuration
    
    Args:
        seed (int, optional): Seed number to set the state of all random number generators
        cudnn_benchmark_enabled (bool, optional): Enable CuDNN benchmark for the sake of performance
        cudnn_deterministic (bool, optional): Make cudnn deterministic (reproducible training)
        
    """
    seed: int = 42
    cudnn_benchmark_enabled: bool = False
    cudnn_deterministic: bool = True
    
    @classmethod
    def from_yaml(cls, config: dict):
        """Load system configuration from a dictionary loaded from YAML.

        Args:
            config (dict): Dictionary with loaded YAML configuration.

        Returns:
            SystemConfig: An instance of the class with values from YAML.
        """
        system_config = config.get('system', {})
        
        return cls(
            seed=system_config.get('seed', cls.seed),
            cudnn_benchmark_enabled=system_config.get('cudnn_benchmark_enabled', cls.cudnn_benchmark_enabled),
            cudnn_deterministic=system_config.get('cudnn_deterministic', cls.cudnn_deterministic)
        )


@dataclass
class DatasetConfig:
    """Class of Data Configuration

    Args:
        root_dir (str, optional): Root directory
        json_file (str, optional): Name of json file of datasets
        train_transforms (torch.Tensor, optional): Data transformation to use during training data preparation
        test_transforms (torch.Tensor, optional): Data transformation to use during test data preparation
        
    """
    root_dir: str = "data"  # dataset directory root
    json_file: str = "datasets.json"
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during training data preparation
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during test data preparation
    
    @classmethod
    def from_yaml(cls, config: dict):
        """Load dataset configuration from a dictionary loaded from YAML.

        Args:
            config (dict): Dictionary with loaded YAML configuration.

        Returns:
            DatasetConfig: An instance of the class with values from YAML.
        """
        dataset_config = config.get('dataset', {})
        
        return cls(
            root_dir=dataset_config.get('root_dir', cls.root_dir),
            json_file=dataset_config.get('json_file', cls.json_file)
        )


@dataclass
class DataloaderConfig:
    """Class of Dataloader Configuration

    Returns:
        batch_size (int, optional): Amount of data to pass through the network at each forward-backward iteration
        num_workers (int, optional): Number of concurrent processes using to prepare data, for free Colab num_workers=2, for free Kaggle num_workers=4
        data_augmentation (bool, optional): Value for setting augmentation method of data
    """
    batch_size: int = 250
    num_workers: int = 2
    data_augmentation: bool = False
    
    @classmethod
    def from_yaml(cls, config: dict):
        """Load dataloader configuration from a dictionary loaded from YAML.

        Args:
            config (dict): Dictionary with loaded YAML configuration.

        Returns:
            DataloaderConfig: An instance of the class with values from YAML.
        """
        
        dataloader_config = config.get('dataloader', {})
        
        return cls(
            batch_size=dataloader_config.get('batch_size', cls.batch_size),
            num_workers=dataloader_config.get('num_workers', cls.num_workers),
            data_augmentation=dataloader_config.get('data_augmentation', cls.data_augmentation)
        )


@dataclass
class OptimizerConfig:
    """Class of Optimizer Configuration

    Args:
        learning_rate (float, optional): Determines the speed of network's weights update
        momentum (float, optional): Used to improve vanilla SGD algorithm and provide better handling of local minimas
        weight_decay (float, optional): Amount of additional regularization on the weights values
        lr_step_milestones (Iterable, optional): At which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
        lr_gamma (float, optional): Multiplier applied to current learning rate at each of lr_ctep_milestones
    """
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    lr_step_milestones: Iterable = (
        30, 40
    )
    lr_gamma: float = 0.1
    
    @classmethod
    def from_yaml(cls, config: dict):
        """Load optimizer configuration from a dictionary loaded from YAML.

        Args:
            config (dict): Dictionary with loaded YAML configuration.

        Returns:
            OptimizerConfig: An instance of the class with values from YAML.
        """
        
        optimizer_config = config.get('optimizer', {})
        
        return cls(
            learning_rate=optimizer_config.get('learning_rate', cls.learning_rate),
            momentum=optimizer_config.get('momentum', cls.momentum),
            weight_decay=optimizer_config.get('weight_decay', cls.weight_decay),
            lr_step_milestones=tuple(optimizer_config.get('lr_step_milestones', cls.lr_step_milestones)),
            lr_gamma=optimizer_config.get('lr_gamma', cls.lr_gamma)
        )


@dataclass
class TrainerConfig:
    """Class of Training Configuration

    Args:
        model_dir (str, optional): Directory to save model states
        model_save_best (bool, optional): flag for save model with best accuracy 
        model_saving_frequency (int, optional): Frequency of model state savings per epochs
        device (str, optional): Device to use for training.
        epoch_num (int, optional): Number of times the whole dataset will be passed through the network
        progress_bar (bool, optional): Enable progress bar visualization during train process    
    """    
    
    model_dir: str = "checkpoints"
    model_save_best: bool = True
    model_saving_frequency: int = 1
    device: str = "cpu"
    epoch_num: int = 50
    progress_bar: bool = False
    
    @classmethod
    def from_yaml(cls, config: dict):
        """Load training configuration from a dictionary loaded from YAML.

        Args:
            config (dict): Dictionary with loaded YAML configuration.

        Returns:
            TrainerConfig: An instance of the class with values from YAML.
        """
        
        trainer_config = config.get('trainer', {})
        
        return cls (
            model_dir=trainer_config.get('model_dir', cls.model_dir),
            model_save_best=trainer_config.get('model_save_best', cls.model_save_best),
            model_saving_frequency=trainer_config.get('model_saving_frequency', cls.model_saving_frequency),
            device=trainer_config.get('device', cls.device),
            epoch_num=trainer_config.get('epoch_num', cls.epoch_num),
            progress_bar=trainer_config.get('progress_bar', cls.progress_bar)
        )

       
def load_config_from_yaml(config_path: str):
    """Function for load YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)
        
    # Creating instances of classes based on the loaded config
    system_config = SystemConfig.from_yaml(config_data)
    dataset_config = DatasetConfig.from_yaml(config_data)
    dataloader_config = DataloaderConfig.from_yaml(config_data)
    optimizer_config = OptimizerConfig.from_yaml(config_data)
    trainer_config = TrainerConfig.from_yaml(config_data)

    return system_config, dataset_config, dataloader_config, optimizer_config, trainer_config
        
    
        
