"""
Learning rate schedulers for PyMC Variational Inference

This module provides learning rate scheduling capabilities for PyMC VI optimization,
based on the PyMC proposals (GitHub issues #6954, #7010).

Date            Programmers                                Descriptions of Change
====         ================                              ======================
2025/08/19      Kianté Fernandez<kiantefernan@gmail.com>   Initial implementation
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
from pymc.variational.callbacks import Callback
import logging

logger = logging.getLogger(__name__)


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when loss has stopped improving.
    
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors the loss
    and if no improvement is seen for a 'patience' number of epochs,
    the learning rate is reduced.
    
    Based on Keras ReduceLROnPlateau
    
    Parameters:
    -----------
    learning_rate : pytensor.shared
        The shared learning rate variable to be modified
    monitor : str, default 'loss'
        Quantity to monitor ('loss')
    factor : float, default 0.1
        Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience : int, default 10
        Number of epochs with no improvement after which learning rate will be reduced
    min_lr : float, default 1e-7
        Lower bound on the learning rate
    cooldown : int, default 0
        Number of epochs to wait before resuming normal operation
    verbose : bool, default True
        If True, print messages when learning rate is reduced
    """
    
    def __init__(self, learning_rate, monitor='loss', factor=0.1, patience=10, 
                 min_lr=1e-7, cooldown=0, verbose=True):
        
        super().__init__()
        
        if not hasattr(learning_rate, 'get_value') or not hasattr(learning_rate, 'set_value'):
            raise ValueError("learning_rate must be a pytensor.shared variable")
        
        self.learning_rate = learning_rate
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.cooldown = cooldown
        self.verbose = verbose
        
        # Internal state
        self.wait = 0
        self.cooldown_counter = 0
        self.best = np.inf
        self.initial_lr = learning_rate.get_value()
        
        # History tracking
        self.lr_history = []
        self.loss_history = []
        
    def __call__(self, approx, loss, i):
        """
        Called at each iteration during VI fitting.
        
        Parameters:
        -----------
        approx : pymc.variational.Approximation
            Current approximation
        loss : array-like
            Loss history array up to current iteration
        i : int
            Current iteration number
        """
        
        # Extract current loss from loss history array
        if hasattr(loss, '__len__') and len(loss) > 0:
            current_loss = float(loss[-1])  # Get the most recent loss
        else:
            current_loss = float(loss)  # Fallback for scalar loss
            
        current_lr = self.learning_rate.get_value()
        
        # Track history
        self.lr_history.append(current_lr)
        self.loss_history.append(current_loss)
        
        # Skip if in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # Check for improvement
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        # Reduce learning rate if no improvement for patience epochs
        if self.wait >= self.patience:
            old_lr = current_lr
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                self.learning_rate.set_value(new_lr)
                if self.verbose:
                    logger.info(f"Iteration {i}: Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                    print(f"Iteration {i}: Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                
                self.wait = 0
                self.cooldown_counter = self.cooldown
                
    def get_lr_history(self):
        """Get the learning rate history."""
        return np.array(self.lr_history)
        
    def get_loss_history(self):
        """Get the loss history."""
        return np.array(self.loss_history)


class StepLRScheduler(Callback):
    """
    Step learning rate scheduler that reduces learning rate at fixed intervals.
    
    Parameters:
    -----------
    learning_rate : pytensor.shared
        The shared learning rate variable to be modified
    step_size : int, default 1000
        Period of learning rate decay (in iterations)
    gamma : float, default 0.1
        Multiplicative factor of learning rate decay
    min_lr : float, default 1e-7
        Lower bound on the learning rate
    verbose : bool, default True
        If True, print messages when learning rate is reduced
    """
    
    def __init__(self, learning_rate, step_size=1000, gamma=0.1, min_lr=1e-7, verbose=True):
        
        super().__init__()
        
        if not hasattr(learning_rate, 'get_value') or not hasattr(learning_rate, 'set_value'):
            raise ValueError("learning_rate must be a pytensor.shared variable")
            
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.initial_lr = learning_rate.get_value()
        self.lr_history = []
        
    def __call__(self, approx, loss, i):
        """Called at each iteration during VI fitting."""
        
        current_lr = self.learning_rate.get_value()
        self.lr_history.append(current_lr)
        
        # Reduce learning rate at step intervals
        if i > 0 and i % self.step_size == 0:
            old_lr = current_lr
            new_lr = max(old_lr * self.gamma, self.min_lr)
            
            if new_lr < old_lr:
                self.learning_rate.set_value(new_lr)
                if self.verbose:
                    logger.info(f"Iteration {i}: Step LR reduction from {old_lr:.2e} to {new_lr:.2e}")
                    print(f"Iteration {i}: Step LR reduction from {old_lr:.2e} to {new_lr:.2e}")
                    
    def get_lr_history(self):
        """Get the learning rate history."""
        return np.array(self.lr_history)


class ExponentialDecayScheduler(Callback):
    """
    Exponential learning rate decay scheduler that applies decay at each iteration.
    
    A smooth decay is often more stable for noisy optimization problems.
    The learning rate is updated at each step as: new_lr = old_lr * gamma
    
    Parameters:
    -----------
    learning_rate : pytensor.shared
        The shared learning rate variable to be modified
    gamma : float
        Multiplicative factor of learning rate decay (e.g., 0.99995). This
        should be very close to 1.0 for a slow, gradual decay.
    min_lr : float, default 1e-10
        Lower bound on the learning rate
    verbose : bool, default False
        If True, print messages periodically to track decay
    """
    
    def __init__(self, learning_rate, gamma, min_lr=1e-10, verbose=False):
        
        super().__init__()
        
        if not hasattr(learning_rate, 'get_value') or not hasattr(learning_rate, 'set_value'):
            raise ValueError("learning_rate must be a pytensor.shared variable")
            
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.initial_lr = learning_rate.get_value()
        self.lr_history = []
        
    def __call__(self, approx, loss, i):
        """Called at each iteration during VI fitting."""
        
        # Always apply the decay
        old_lr = self.learning_rate.get_value()
        new_lr = max(old_lr * self.gamma, self.min_lr)
        
        # Set the new value
        self.learning_rate.set_value(new_lr)
        self.lr_history.append(new_lr)
        
        # Optional: Log the change periodically to avoid spamming the console
        if self.verbose and i > 0 and i % 5000 == 0:
            logger.info(f"Iteration {i}: LR decayed to {new_lr:.2e}")
            print(f"Iteration {i}: LR decayed to {new_lr:.2e}")
                    
    def get_lr_history(self):
        """Get the learning rate history."""
        return np.array(self.lr_history)


def create_scheduler(scheduler_type, learning_rate, **kwargs):
    """
    Function to create learning rate schedulers.
    
    Parameters:
    -----------
    scheduler_type : str
        Type of scheduler: 'plateau', 'step', 'exponential'
    learning_rate : pytensor.shared
        The shared learning rate variable
    **kwargs : dict
        Additional parameters for the specific scheduler
        
    Returns:
    --------
    Callback
        The configured scheduler callback
    """
    
    if scheduler_type.lower() in ['plateau', 'reduce_on_plateau']:
        return ReduceLROnPlateau(learning_rate, **kwargs)
    elif scheduler_type.lower() == 'step':
        return StepLRScheduler(learning_rate, **kwargs)
    elif scheduler_type.lower() in ['exponential', 'exp']:
        return ExponentialDecayScheduler(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Available options: 'plateau', 'step', 'exponential'")