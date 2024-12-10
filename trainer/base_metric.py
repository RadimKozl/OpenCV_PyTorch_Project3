#!/usr/bin/python3
"""Base Metric Class module

Abstract Base class for Metrics Class
"""

# Imports libraries
from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """Abstract class of Base metrics

    Args:
        ABC (class): Abstract class constructor
    """
    
    @abstractmethod
    def update_value(self, pred: Any, target: Any) -> Any:
        """Abstract method for update value

        Args:
            pred (Any): predicted values
            target (Any): target values

        Returns:
            Any: updated value
        """
        pass

    @abstractmethod
    def get_metric_value(self) -> Any:
        """Abstract method returns metric value

        Returns:
            Any: metric value
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Abstract method reset metric value
        """
        pass
