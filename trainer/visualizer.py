#!/usr/bin/python3

"""Visualizer Base Class module
"""

# Import libraries
from abc import ABC, abstractmethod
from typing import Any


class Visualizer(ABC):
    """Abstract class of Visualizer Base class

    Args:
        LogSetting (class): Abstract class 
    """
    @abstractmethod
    def update_charts(
        self,
        train_metric: Any,
        train_loss: Any,
        test_metric: Any,
        test_loss: Any,
        learning_rate: float,
        epoch: int
    ) -> Any:
        """Update method

        Args:
            train_metric (Any): metric of training of model
            train_loss (Any): loss of training of model
            test_metric (Any): metrict of test of model
            test_loss (Any): loss of test of model
            learning_rate (float): learning rate value
            epoch (int): number of epoch
        Returns:
            Any: updated values
        """
        pass
