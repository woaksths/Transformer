import torch
import torch.nn as nn
import itertools

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate 
        self.optimizer.step()
    
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step**(-0.5), step*self.warmup**(-1.5)))                              


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class Optimizer(object):
    """ The Optimizer class encapsulates torch.optim package and provides functionalities
    for learning rate scheduling and gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.SGD(params)
        max_grad_norm (float, optional): value used for gradient norm clipping,
            set 0 to disable (default 0)
    """

    _ARG_MAX_GRAD_NORM = 'max_grad_norm'

    def __init__(self, optim, max_grad_norm=0):
        self.optimizer = optim
        self.scheduler = None
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, scheduler):
        """ Set the learning rate scheduler.

        Args:
            scheduler (torch.optim.lr_scheduler.*): object of learning rate scheduler,
               e.g. torch.optim.lr_scheduler.StepLR
        """
        self.scheduler = scheduler

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary. """
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

    def update(self, loss, epoch):
        """ Update the learning rate if the criteria of the scheduler are met.

        Args:
            loss (float): The current loss.  It could be training loss or developing loss
                depending on the caller.  By default the supervised trainer uses developing
                loss.
            epoch (int): The current epoch number.
        """
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()
