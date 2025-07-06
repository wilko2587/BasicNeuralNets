"""
Dataset handling for Deep Set Autoencoder

This module provides the SetDataset class and related utilities for loading
and processing set-based data for the Deep Set autoencoder.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SetDataset(Dataset):
    """
    Dataset that handles sets of points from multiple distributions.
    Each set contains points from 3 different distributions.
    """
    
    def __init__(self, data, labels):
        """
        Initialize dataset with pre-generated data.
        
        Args:
            data: List of torch tensors, each of shape (set_size, input_dim)
            labels: List of integers indicating the primary distribution
        """
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataloaders(data, labels, batch_size=32, train_split=0.7, val_split=0.15, 
                      shuffle=True, num_workers=0):
    """
    Create train, validation, and test dataloaders from data and labels.
    
    Args:
        data: List of torch tensors, each of shape (set_size, input_dim)
        labels: List of integers indicating the primary distribution
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Calculate split indices
    n_samples = len(data)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    # Split data
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    val_data = data[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    
    # Create datasets
    train_dataset = SetDataset(train_data, train_labels)
    val_dataset = SetDataset(val_data, val_labels)
    test_dataset = SetDataset(test_data, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader 