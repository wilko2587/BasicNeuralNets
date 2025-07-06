"""
Loss Functions for Deep Set Autoencoder

This module provides the Chamfer distance loss function and related utilities
for training the Deep Set autoencoder.
"""

import torch
import torch.nn as nn


def chamfer_distance(set1, set2, reduction='mean'):
    """
    Compute the Chamfer distance between two sets of points.
    
    Args:
        set1: Tensor of shape (batch_size, n_points1, dim)
        set2: Tensor of shape (batch_size, n_points2, dim)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Chamfer distance (scalar if reduction is 'mean' or 'sum')
    """
    batch_size = set1.shape[0]
    
    # Compute pairwise distances between all points in set1 and set2
    # set1: (batch_size, n_points1, 1, dim)
    # set2: (batch_size, 1, n_points2, dim)
    set1_expanded = set1.unsqueeze(2)
    set2_expanded = set2.unsqueeze(1)
    
    # Squared Euclidean distances
    distances = torch.sum((set1_expanded - set2_expanded) ** 2, dim=-1)
    
    # For each point in set1, find nearest point in set2
    min_distances_1to2, _ = torch.min(distances, dim=2)
    
    # For each point in set2, find nearest point in set1
    min_distances_2to1, _ = torch.min(distances, dim=1)
    
    # Chamfer distance is the sum of both directions
    chamfer_dist = torch.mean(min_distances_1to2, dim=1) + torch.mean(min_distances_2to1, dim=1)
    
    if reduction == 'mean':
        return torch.mean(chamfer_dist)
    elif reduction == 'sum':
        return torch.sum(chamfer_dist)
    else:
        return chamfer_dist


class ChamferLoss(nn.Module):
    """Chamfer distance loss for set reconstruction."""
    
    def __init__(self, reduction='mean'):
        super(ChamferLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_set, target_set):
        """
        Args:
            pred_set: Predicted set, shape (batch_size, set_size, dim)
            target_set: Target set, shape (batch_size, set_size, dim)
        
        Returns:
            Chamfer distance loss
        """
        return chamfer_distance(pred_set, target_set, self.reduction) 