"""
Deep Set Autoencoder Model

This module implements the Deep Set autoencoder architecture that can handle
both numerical and categorical features in set-based data.
"""

import torch
import torch.nn as nn


class DeepSetEncoder(nn.Module):
    """Deep Set Encoder that maps a set of vectors to a single embedding."""
    
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64, 
                 num_layers=3, aggregation='mean'):
        super(DeepSetEncoder, self).__init__()
        self.aggregation = aggregation
        
        # φ network: processes each element in the set
        phi_layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        
        for i in range(len(dims) - 1):
            phi_layers.append(nn.Linear(dims[i], dims[i + 1]))
            phi_layers.append(nn.ReLU())
            phi_layers.append(nn.BatchNorm1d(dims[i + 1]))
            
        self.phi = nn.Sequential(*phi_layers)
        
        # ρ network: processes the aggregated representation
        rho_layers = []
        rho_dims = [hidden_dim] + [hidden_dim] * (num_layers - 1) + [embedding_dim]
        
        for i in range(len(rho_dims) - 1):
            rho_layers.append(nn.Linear(rho_dims[i], rho_dims[i + 1]))
            if i < len(rho_dims) - 2:  # No activation after last layer
                rho_layers.append(nn.ReLU())
                rho_layers.append(nn.BatchNorm1d(rho_dims[i + 1]))
                
        self.rho = nn.Sequential(*rho_layers)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, set_size, input_dim)
        Returns:
            embedding: Tensor of shape (batch_size, embedding_dim)
        """
        batch_size, set_size, input_dim = x.shape
        
        # Reshape to process all set elements at once
        x_flat = x.view(-1, input_dim)
        
        # Apply φ to each element
        phi_out = self.phi(x_flat)
        
        # Reshape back to (batch_size, set_size, hidden_dim)
        phi_out = phi_out.view(batch_size, set_size, -1)
        
        # Aggregate across the set dimension
        if self.aggregation == 'mean':
            aggregated = torch.mean(phi_out, dim=1)
        elif self.aggregation == 'sum':
            aggregated = torch.sum(phi_out, dim=1)
        elif self.aggregation == 'max':
            aggregated, _ = torch.max(phi_out, dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Apply ρ to get final embedding
        embedding = self.rho(aggregated)
        
        return embedding


class DeepSetDecoder(nn.Module):
    """Deep Set Decoder that reconstructs a set from an embedding."""
    
    def __init__(self, embedding_dim, output_dim, set_size, 
                 hidden_dim=128, num_layers=3):
        super(DeepSetDecoder, self).__init__()
        self.set_size = set_size
        
        # Network to generate set elements from embedding
        layers = []
        dims = [embedding_dim] + [hidden_dim] * num_layers + [set_size * output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                
        self.decoder = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def forward(self, embedding):
        """
        Args:
            embedding: Tensor of shape (batch_size, embedding_dim)
        Returns:
            reconstructed_set: Tensor of shape (batch_size, set_size, output_dim)
        """
        batch_size = embedding.shape[0]
        
        # Generate flattened set
        output = self.decoder(embedding)
        
        # Reshape to set format
        reconstructed_set = output.view(batch_size, self.set_size, self.output_dim)
        
        return reconstructed_set


class DeepSetAutoencoder(nn.Module):
    """Complete Deep Set Autoencoder."""
    
    def __init__(self, input_dim, embedding_dim=64, set_size=10,
                 hidden_dim=128, num_layers=3, aggregation='mean',
                 categorical_dims=None, embedding_sizes=None):
        super(DeepSetAutoencoder, self).__init__()
        
        self.categorical_dims = categorical_dims or []
        self.numerical_dim = input_dim - len(self.categorical_dims)
        
        # Create embeddings for categorical features
        self.embeddings = nn.ModuleList()
        total_embedding_dim = self.numerical_dim
        
        if categorical_dims and embedding_sizes:
            for cat_dim, emb_size in zip(categorical_dims, embedding_sizes):
                self.embeddings.append(nn.Embedding(cat_dim, emb_size))
                total_embedding_dim += emb_size
        
        # Encoder and Decoder
        self.encoder = DeepSetEncoder(
            total_embedding_dim, hidden_dim, embedding_dim, 
            num_layers, aggregation
        )
        self.decoder = DeepSetDecoder(
            embedding_dim, input_dim, set_size, 
            hidden_dim, num_layers
        )
        
    def embed_categorical(self, x):
        """Embed categorical features if present."""
        if not self.categorical_dims:
            return x
        
        batch_size, set_size, _ = x.shape
        
        # Split numerical and categorical features
        numerical = x[:, :, :self.numerical_dim]
        embedded_parts = [numerical]
        
        cat_start = self.numerical_dim
        for i, (cat_dim, embedding) in enumerate(zip(self.categorical_dims, self.embeddings)):
            cat_feature = x[:, :, cat_start].long()
            embedded = embedding(cat_feature)
            embedded_parts.append(embedded)
            cat_start += 1
        
        # Concatenate all parts
        return torch.cat(embedded_parts, dim=-1)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, set_size, input_dim)
        Returns:
            reconstructed: Tensor of shape (batch_size, set_size, input_dim)
            embedding: Tensor of shape (batch_size, embedding_dim)
        """
        # Embed categorical features if present
        x_embedded = self.embed_categorical(x)
        
        # Encode
        embedding = self.encoder(x_embedded)
        
        # Decode
        reconstructed = self.decoder(embedding)
        
        return reconstructed, embedding
    
    def encode(self, x):
        """Only encode the input set."""
        x_embedded = self.embed_categorical(x)
        return self.encoder(x_embedded)
    
    def decode(self, embedding):
        """Only decode from embedding."""
        return self.decoder(embedding) 