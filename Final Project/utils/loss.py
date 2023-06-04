import torch
import torch.nn.functional as F
import torch.nn as nn


def get_loss(loss_name, params=None):
    if loss_name=='bce':
        criteriation = nn.BCELoss()
        
    elif loss_name=='focal':
        alpha, gamma = params
        criteriation = FocalLoss(alpha=alpha, gamma=gamma)
        
    elif loss_name=='w':
        criteriation = WeightedMultilabel(params)
        
    else:
        raise ValueError('Incorrect loss')
    
    return criteriation



class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.cross_entropy(inputs, targets)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss
    
    
class WeightedMultilabel(nn.Module):  
    def __init__(self, weights: torch.Tensor):  
        super(WeightedMultilabel, self).__init__()  
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')  
        self.weights = weights  
  
    def forward(self, outputs, targets):  
        loss = self.cerition(outputs, targets)  
        return (loss * self.weights).mean()  
