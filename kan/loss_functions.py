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
        softmax_probs = torch.softmax(input, dim=1)  # Get probabilities
        weighted_indices = torch.arange(input.size(1), device=input.device, dtype=torch.float32)
        input_expectation = torch.sum(softmax_probs * weighted_indices, dim=1)  # Differentiable expectation

        #input_squeeze = torch.argmax(input, dim=1) # torch.argmax(torch.softmax(self.model(new_data), dim=1), dim=1).detach().numpy()
        #print(input_expectation)
        #print(target)
        #if(self.rooted):
        #    return np.sqrt(super().forward(input.type(torch.float64), target.type(torch.long)))
        #return lambda x, y: torch.mean((x - y) ** 2)
        return torch.mean((input_expectation - target) ** 2)
        #return super().forward(input_squeeze.type(torch.float64), target.type(torch.long))
