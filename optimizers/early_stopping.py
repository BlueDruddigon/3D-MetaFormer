from typing import Callable, Union

import torch


class EarlyStopping:
    """Early Stopping Callback simple implementation"""
    
    mode_dict = {'min': torch.lt, 'max': torch.gt}
    
    def __init__(self, patience: int = 5, mode: str = 'min', min_delta: float = 1e-5) -> None:
        """Early Stopping Callback

        :param patience: number of checks with no improvement.
        :param mode: choices: 'min', 'max'.
            In 'min' mode, training will stop when the quantity monitored has stopped decreasing.
            And in 'max' mode, it will stop when the quantity monitored has stopped increasing.
        :param min_delta: a minimum value for changes in the monitored quantity to qualify as an improvement.
        """
        super().__init__()
        
        self.patience = patience
        self.min_delta = torch.tensor(min_delta) if not isinstance(min_delta, torch.Tensor) else min_delta
        self.counter = 0
        
        self.mode = mode
        if self.mode not in self.mode_dict:
            raise NotImplementedError
        
        self.best_score = torch.tensor(torch.inf) if self.mode == 'min' else torch.tensor(-torch.inf)
    
    @property
    def monitor_op(self) -> Callable:
        """ Return monitor operation from mode """
        return self.mode_dict[self.mode]
    
    def step(self, metric: Union[float, torch.Tensor]) -> bool:
        """Monitor the quantity

        :param metric: the monitored quantity metric value.
        :return: a boolean flag that provides information if the training needs early stopped
        """
        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric)
        if self.monitor_op(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
        elif self.monitor_op(self.min_delta, metric - self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
