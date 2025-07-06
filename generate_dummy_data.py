"""
Dummy Data Generator for Deep Set Autoencoder

This module generates synthetic set-based data for training and testing
the Deep Set autoencoder. Each set contains points from 3 different distributions.
"""

import torch
import numpy as np


def generate_dummy_data(num_sets=1000, set_size=20, input_dim=5, 
                       categorical_features=0, num_categories=5, seed=42):
    """
    Generate dummy data for Deep Set autoencoder training.
    
    Args:
        num_sets: Number of sets to generate
        set_size: Number of points in each set
        input_dim: Total dimensionality of each point
        categorical_features: Number of categorical features
        num_categories: Number of categories for each categorical feature
        seed: Random seed for reproducibility
        
    Returns:
        data: List of torch tensors, each of shape (set_size, input_dim)
        labels: List of integers indicating the primary distribution
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    numerical_features = input_dim - categorical_features
    
    # Define 3 distributions with different centers
    dist_centers = [
        np.array([0.0] * numerical_features),
        np.array([5.0] * numerical_features),
        np.array([-5.0] + [5.0] * (numerical_features - 1))
    ]
    dist_stds = [1.0, 1.2, 0.8]  # Different variances for each distribution
    
    data = []
    labels = []
    
    for _ in range(num_sets):
        # Randomly choose which distribution this set primarily belongs to
        primary_dist = np.random.randint(0, 3)
        labels.append(primary_dist)
        
        # Determine how many points from each distribution
        # Primary distribution gets 60-80% of points
        primary_points = int(set_size * np.random.uniform(0.6, 0.8))
        remaining = set_size - primary_points
        
        # Split remaining points between other two distributions
        other_dists = [i for i in range(3) if i != primary_dist]
        split = np.random.randint(0, remaining + 1)
        points_per_dist = {
            primary_dist: primary_points,
            other_dists[0]: split,
            other_dists[1]: remaining - split
        }
        
        # Generate points for this set
        set_points = []
        
        for dist_idx in range(3):
            n_points = points_per_dist[dist_idx]
            if n_points > 0:
                # Generate numerical features
                numerical_data = np.random.normal(
                    dist_centers[dist_idx],
                    dist_stds[dist_idx],
                    size=(n_points, numerical_features)
                )
                
                # Generate categorical features if needed
                if categorical_features > 0:
                    # Each distribution has preference for certain categories
                    cat_probs = np.zeros((3, categorical_features, num_categories))
                    for i in range(3):
                        for j in range(categorical_features):
                            # Create peaked distribution for each cluster
                            probs = np.ones(num_categories) * 0.1
                            probs[(i + j) % num_categories] = 0.6
                            probs = probs / probs.sum()
                            cat_probs[i, j] = probs
                    
                    categorical_data = []
                    for j in range(categorical_features):
                        cats = np.random.choice(
                            num_categories, 
                            size=n_points,
                            p=cat_probs[dist_idx, j]
                        )
                        categorical_data.append(cats)
                    
                    categorical_data = np.array(categorical_data).T
                    
                    # Combine numerical and categorical
                    point_data = np.concatenate([numerical_data, categorical_data], axis=1)
                else:
                    point_data = numerical_data
                
                set_points.extend(point_data)
        
        # Shuffle points within the set to ensure permutation invariance
        set_points = np.array(set_points)
        np.random.shuffle(set_points)
        
        data.append(torch.FloatTensor(set_points))
    
    return data, labels


def get_feature_info(input_dim, categorical_features, num_categories):
    """
    Get feature information for model construction.
    
    Args:
        input_dim: Total input dimension
        categorical_features: Number of categorical features
        num_categories: Number of categories per categorical feature
        
    Returns:
        categorical_dims: List of category counts for each categorical feature
        embedding_sizes: List of embedding dimensions for each categorical feature
    """
    if categorical_features > 0:
        categorical_dims = [num_categories] * categorical_features
        # Simple heuristic for embedding sizes
        embedding_sizes = [min(50, (cat_dim + 1) // 2) for cat_dim in categorical_dims]
        return categorical_dims, embedding_sizes
    return None, None 