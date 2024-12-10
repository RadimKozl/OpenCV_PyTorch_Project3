#!/usr/bin/python3

"""Evalution Metrics Class module
"""

# Import libraries
import torch

from .utils import AverageMeter
from .voc_eval import VOCEvaluator
from .base_metric import BaseMetric


class AccuracyEstimator(BaseMetric):
    """Class of Accuracy Estimator
    
    Args:
        BaseMetric (class): Abstract Base metric class
    """
    def __init__(self, topk=(1, )):
        """Init method of class

        Args:
            topk (tuple, optional): number of values for calculation of precision. Defaults to (1, ).
        """        
        self.topk = topk
        self.metrics = [AverageMeter() for i in range(len(topk) + 1)]

    def reset(self):
        """Reset method
        """ 
        for i in range(len(self.metrics)):
            self.metrics[i].reset()

    def update_value(self, pred, target):
        """Method for Compute precision@k for given values of k

        Args:
            pred (numpy.array): predicted values of model
            target (numpy.array): real labeled values of samples
        """
        
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = target.size(0)

            _, pred = pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i, k in enumerate(self.topk):
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                self.metrics[i].update(correct_k.mul_(100.0 / batch_size).item())

    def get_metric_value(self):
        """Method for return metric 
        Returns:
            dict: return metric values
        """ 
        
        metrics = {}
        for i, k in enumerate(self.topk):
            metrics["top{}".format(k)] = self.metrics[i].avg
        return metrics


class APEstimator(BaseMetric):
    """Class for AP Estimator

    Args:
        BaseMetric (class): Abstract Base metric class
    """   
    def __init__(self, classes):
        self.classes = classes
        self.metrics = AverageMeter()
        self.evaluator = None

    def reset(self):
        """Reset method
        """        
        self.metrics.reset()
        self.evaluator = VOCEvaluator(self.classes)

    def update_value(self, pred, target):
        """Method for Computes AP

        Args:
            pred (numpy.array): predicted values of model
            target (numpy.array): real labeled values of samples
        """        
        
        self.evaluator.add_sample(pred, target)

    def calculate_value(self):
        """Method for Computes AP
        
        Args:
            pred (numpy.array): predicted values of model
            target (numpy.array): real labeled values of samples
        """
        aps = self.evaluator.evaluate()
        for class_idx in range(len(self.classes)):
            if self.classes[class_idx] == '__background__':
                continue
            if len(aps) > class_idx - 1:
                self.metrics.update(aps[class_idx - 1])
            else:
                self.metrics.update(0.0)

    def get_metric_value(self):
        """Method for return metric 
        Returns:
            dict: return metric values
        """ 
        
        metrics = {}
        metrics["mAP"] = self.metrics.avg
        return metrics
