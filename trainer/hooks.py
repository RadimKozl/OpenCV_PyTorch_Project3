#!/usr/bin/python3

"""Module of Hooks for Trainer Class

Implementation of several hooks that used in a Trainer class.
This module implements several hooks (helper functions) for the Trainer class.
This module has the following method implemented:
    - train_hook_default()
    - test_hook_default()
    - end_epoch_hook_classification()
and class:
    - IteratorWithStorage
"""

# Import libraries
from operator import itemgetter

import torch

from tqdm.auto import tqdm

from .utils import AverageMeter, progress_bar


def train_hook_faster_rcnn(
    model,
    loader,
    optimizer,
    device,
    data_getter=None,  
    target_getter=None, 
    iterator_type=progress_bar,
    prefix="",
    stage_progress=False
):
    """ Default train loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            loss: average loss.
    """
    model = model.to(device)
    model = model.train()
    iterator = iterator_type(loader, total=len(loader), prefix=prefix) if stage_progress else loader
    loss_avg = AverageMeter()
    for i, sample in enumerate(iterator):
        
        optimizer.zero_grad()
        
        images = list(image.to(device) for image in sample[0])
        targets = [{key: value.to(device) for key, value in target.items()} for target in sample[1]]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()

        optimizer.step()
        loss_avg.update(losses.item())
        
        # Print progress status if using custom progress bar
        if stage_progress:
            status = (
                f"{prefix}[Train][{i}] Loss_avg: {loss_avg.avg:.5f}, Loss: {loss_avg.val:.5f}, "
                f"loss_box_reg: {loss_dict['loss_box_reg'].item():.5f}, "
                f"loss_classifier: {loss_dict['loss_classifier'].item():.5f}, "
                f"loss_objectness: {loss_dict['loss_objectness'].item():.5f}, "
                f"loss_rpn_box_reg: {loss_dict['loss_rpn_box_reg'].item():.5f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}"
            )
            print(status, end="\r")
        
        del images, targets, loss_dict
        torch.cuda.empty_cache()
        
    if stage_progress:
        print()  # Ensure the final line of the progress bar is cleared
        
    return {"loss": loss_avg.avg}


def train_hook_default(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    data_getter=itemgetter("image"),
    target_getter=itemgetter("mask"),
    iterator_type=tqdm,
    prefix="",
    stage_progress=False
):
    """ Default train loop function.

    Args:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            loss: average loss.
    """
    model = model.train()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    for i, sample in enumerate(iterator):
        optimizer.zero_grad()
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        predicts = model(inputs)
        loss = loss_fn(predicts, targets)
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())
        status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, LR: {4:.5}".format(
            prefix, i, loss_avg.avg, loss_avg.val, optimizer.param_groups[0]["lr"]
        )
        iterator.set_description(status)
        # Freeing up memory
        del inputs, targets, predicts
        torch.cuda.empty_cache()
        
    return {"loss": loss_avg.avg}


def test_hook_faster_rcnn(
    model,
    loader,
    metric_fn,
    device,
    data_getter=None,  
    target_getter=None,
    iterator_type=progress_bar,
    prefix="",
    stage_progress=False,
    get_key_metric=itemgetter("AP")
):
    """ Default test loop function.

    Arguments:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        metric_fn (callable): evaluation metric function.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            metric: output metric.
            loss: average loss.
    """
    model = model.to(device)
    model = model.eval()
    iterator = iterator_type(loader, total=len(loader), prefix=prefix) if stage_progress else loader
    metric_fn.reset()

    for i, sample in enumerate(iterator):
        
        image = list(image.to(device) for image in sample[0])
        targets = [{key: value.to(device) for key, value in target.items()} for target in sample[1]]
        
        eval_targets = []
        
        for target in targets:
            bboxes = target['boxes']
            labels = target['labels']
            img_targets = torch.empty((len(labels), 5))
            img_targets[:, :4] = bboxes
            img_targets[:, 4] = labels
            eval_targets.append(img_targets)
                   
            
        with torch.no_grad():
            detections = model(image)
            
        predictions = []
        for dets in detections:
            bboxes = dets['boxes']
            labels = dets['labels']
            scores = dets['scores']
            img_det = torch.empty((len(labels), 6))
            img_det[:, :4] = bboxes
            img_det[:, 4] = scores
            img_det[:, 5] = labels
            predictions.append(img_det)
            
        for det, target in zip(predictions, eval_targets):
            metric_fn.update_value(det, target)
        
        # Print status manually for custom progress bar
        if stage_progress and get_key_metric is not None:
            status = f"{prefix}[Test][{i}] Metric_avg: {get_key_metric(metric_fn.get_metric_value()):.5f}"
            print(status, end="\r")
            
        
    metric_fn.calculate_value()
    if stage_progress:
        print()  # Ensure a clean final line after progress
        
    output = {"metric": metric_fn.get_metric_value()}
    return output


def test_hook_default(
    model,
    loader,
    loss_fn,
    metric_fn,
    device,
    data_getter=itemgetter("image"),
    target_getter=itemgetter("mask"),
    iterator_type=tqdm,
    prefix="",
    stage_progress=False,
    get_key_metric=itemgetter("accuracy")
):
    """ Default test loop function.

    Args:
        model (nn.Module): torch model which will be train.
        loader (torch.utils.DataLoader): dataset loader.
        loss_fn (callable): loss function.
        metric_fn (callable): evaluation metric function.
        device (str): Specifies device at which samples will be uploaded.
        data_getter (Callable): function object to extract input data from the sample prepared by dataloader.
        target_getter (Callable): function object to extract target data from the sample prepared by dataloader.
        iterator_type (iterator): type of the iterator.
        prefix (string): prefix which will be add to the description string.
        stage_progress (bool): if True then progress bar will be show.

    Returns:
        Dictionary of output metrics with keys:
            metric: output metric.
            loss: average loss.
    """
    model = model.eval()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    metric_fn.reset()
    for i, sample in enumerate(iterator):
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        with torch.no_grad():
            predict = model(inputs)
            loss = loss_fn(predict, targets)
        loss_avg.update(loss.item())
        predict = predict.softmax(dim=1).detach()
        metric_fn.update_value(predict, targets)
        status = "{0}[Test][{1}] Loss_avg: {2:.5}".format(prefix, i, loss_avg.avg)
        if get_key_metric is not None:
            status = status + ", Metric_avg: {0:.5}".format(get_key_metric(metric_fn.get_metric_value()))
        iterator.set_description(status)
        # Freeing up memory
        del inputs, targets, predict
        torch.cuda.empty_cache()
        
    output = {"metric": metric_fn.get_metric_value(), "loss": loss_avg.avg}
    return output


def end_epoch_hook_faster_rcnn(iterator, epoch, output_train, output_test):
    """ Default end_epoch_hook for detection tasks.
    Arguments:
        iterator (iter): iterator.
        epoch (int): number of epoch to store.
        output_train (dict): description of the train stage.
        output_test (dict): description of the test stage.
        trainer (Trainer): trainer object.
    """
    if hasattr(iterator, "set_description"):
        iterator.set_description(
            "epoch: {0}, test_AP: {1:.5}, train_loss: {2:.5}".format(
                epoch, output_test["metric"]["mAP"], output_train["loss"]
            )
        )


class IteratorWithStorage(tqdm):
    """ Class to store logs of deep learning experiments."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None

    def init_metrics(self, keys=None):
        """ Initialise metrics list.

        Arguments:
            keys (list): list of available keys that need to store.
        """
        if keys is None:
            keys = []
        self.metrics = {k: [] for k in keys}

    def get_metrics(self):
        """ Get stored metrics.

        Returns:
            Dictionary of stored metrics.
        """
        return self.metrics

    def reset_metrics(self):
        """ Reset stored metrics. """
        for key, _ in self.metrics.items():
            self.metrics[key] = []

    def set_description(self, desc=None, refresh=True):
        """ Set description which will be view in status bar.

        Arguments:
            desc (str, optional): description of the iteration.
            refresh (bool): refresh description.
        """
        self.desc = desc or ''
        if self.metrics is not None:
            self._store_metrics(desc)
        if refresh:
            self.refresh()

    def set_extra_description(self, key, val):
        """ Set extra description which will not be view in status bar.

        Arguments:
            key (str): key of the extra description.
            val (str): value of the extra description.
        """
        if self.metrics is not None and key in self.metrics:
            self.metrics[key] = val

    def _store_metrics(self, format_string):
        metrics = dict(x.split(": ") for x in format_string.split(", "))
        for key, val in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(float(val))
