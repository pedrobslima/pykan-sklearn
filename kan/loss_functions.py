import torch
import numpy as np

class kanCELoss(torch.nn.CrossEntropyLoss): # It works for any amount of classes
  def forward(self, input, target):
        return super().forward(input.type(torch.float64), target.type(torch.long))

class kanMSELoss(torch.nn.MSELoss):
    #def __init__(self, size_average=None, reduce=None, reduction = 'mean', rooted=False):
    #    super().__init__(size_average, reduce, reduction)
    #    self.rooted = rooted
    def forward(self, input, target):
        #if(self.rooted):
        #    return np.sqrt(super().forward(input.type(torch.float64), target.type(torch.long)))
        return super().forward(input.type(torch.float64), target.type(torch.long))
