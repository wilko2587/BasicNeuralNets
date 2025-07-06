"""
Main Script for Deep Set Autoencoder

This script demonstrates the complete pipeline:
1. Generate dummy data in memory
2. Initialize dataset and dataloaders
3. Initialize model
4. Train the model
5. Visualize results
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from itertools import permutations
from tqdm import tqdm

from .generate_dummy_data import generate_dummy_data, get_feature_info
from .dataset import create_dataloaders
from .model import DeepSetAutoencoder
from .scorer import ChamferLoss


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(device)
            
            # Forward pass
            reconstructed, embedding = model(data)
            loss = criterion(reconstructed, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for batch_idx, (data, labels) in enumerate(pbar):
                data = data.to(device)
                
                # Forward pass
                reconstructed, embedding = model(data)
                loss = criterion(reconstructed, data)
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def extract_embeddings(model, dataloader, device):
    """Extract embeddings and labels from the model."""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            embedding = model.encode(data)
            embeddings.append(embedding.cpu().numpy())
            labels.extend(label.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    
    return embeddings, labels


def visualize_embeddings(embeddings, labels, title="Deep Set Embeddings"):
    """Visualize embeddings using PCA."""
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Define colors for each cluster
    colors = ['red', 'blue', 'green']
    cluster_names = ['Distribution 1', 'Distribution 2', 'Distribution 3']
    
    # Plot each cluster
    for i in range(3):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=colors[i], label=cluster_names[i], alpha=0.6, s=50)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_embeddings(embeddings, labels):
    """Analyze the quality of embeddings."""
    
    # Calculate silhouette score
    silhouette = silhouette_score(embeddings, labels)
    print(f"\nSilhouette Score: {silhouette:.4f}")
    
    # Perform K-means clustering and compare with true labels
    kmeans = KMeans(n_clusters=3, random_state=42)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    # Calculate clustering accuracy (best permutation)
    best_acc = 0
    for perm in permutations(range(3)):
        mapping = {i: perm[i] for i in range(3)}
        mapped_pred = [mapping[p] for p in predicted_labels]
        acc = np.mean(mapped_pred == labels)
        best_acc = max(best_acc, acc)
    
    print(f"K-means Clustering Accuracy: {best_acc:.4f}")
    
    # Calculate inter-cluster and intra-cluster distances
    cluster_centers = []
    intra_distances = []
    
    for i in range(3):
        mask = labels == i
        cluster_embeddings = embeddings[mask]
        center = np.mean(cluster_embeddings, axis=0)
        cluster_centers.append(center)
        
        # Intra-cluster distance
        distances = np.sqrt(np.sum((cluster_embeddings - center) ** 2, axis=1))
        intra_distances.append(np.mean(distances))
    
    print(f"\nAverage Intra-cluster Distances:")
    for i in range(3):
        print(f"  Cluster {i}: {intra_distances[i]:.4f}")
    
    print(f"\nInter-cluster Distances:")
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.sqrt(np.sum((cluster_centers[i] - cluster_centers[j]) ** 2))
            print(f"  Cluster {i} <-> Cluster {j}: {dist:.4f}")


def main():
    """Main function to run the complete pipeline."""
    
    print("=" * 60)
    print("DEEP SET AUTOENCODER - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_sets': 5000,
        'set_size': 20,
        'input_dim': 5,
        'categorical_features': 0,
        'num_categories': 5,
        'embedding_dim': 32,
        'hidden_dim': 128,
        'num_layers': 3,
        'aggregation': 'mean',
        'batch_size': 64,
        'learning_rate': 1e-3,
        'num_epochs': 30,
        'seed': 42
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 1. Generate dummy data in memory
    print(f"\n{'='*40}")
    print("1. GENERATING DUMMY DATA")
    print(f"{'='*40}")
    
    data, labels = generate_dummy_data(
        num_sets=config['num_sets'],
        set_size=config['set_size'],
        input_dim=config['input_dim'],
        categorical_features=config['categorical_features'],
        num_categories=config['num_categories'],
        seed=config['seed']
    )
    
    print(f"Generated {len(data)} sets with {config['set_size']} points each")
    print(f"Input dimension: {config['input_dim']}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # 2. Initialize dataset and dataloaders
    print(f"\n{'='*40}")
    print("2. INITIALIZING DATASET")
    print(f"{'='*40}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data=data,
        labels=labels,
        batch_size=config['batch_size'],
        train_split=0.7,
        val_split=0.15,
        shuffle=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 3. Initialize model
    print(f"\n{'='*40}")
    print("3. INITIALIZING MODEL")
    print(f"{'='*40}")
    
    categorical_dims, embedding_sizes = get_feature_info(
        config['input_dim'],
        config['categorical_features'],
        config['num_categories']
    )
    
    model = DeepSetAutoencoder(
        input_dim=config['input_dim'],
        embedding_dim=config['embedding_dim'],
        set_size=config['set_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        aggregation=config['aggregation'],
        categorical_dims=categorical_dims,
        embedding_sizes=embedding_sizes
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = ChamferLoss()
    
    # 5. Train the model
    print(f"\n{'='*40}")
    print("4. TRAINING MODEL")
    print(f"{'='*40}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with val loss: {val_loss:.4f}")
    
    # 6. Test final model
    print(f"\n{'='*40}")
    print("5. TESTING MODEL")
    print(f"{'='*40}")
    
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # 7. Extract and analyze embeddings
    print(f"\n{'='*40}")
    print("6. ANALYZING EMBEDDINGS")
    print(f"{'='*40}")
    
    embeddings, labels = extract_embeddings(model, test_loader, device)
    print(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Analyze embeddings
    analyze_embeddings(embeddings, labels)
    
    # 8. Visualize embeddings
    print(f"\n{'='*40}")
    print("7. VISUALIZING EMBEDDINGS")
    print(f"{'='*40}")
    
    visualize_embeddings(embeddings, labels, "Deep Set Autoencoder - Learned Clusters")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print("Generated files:")
    print("- best_model.pth: Trained model weights")
    print("- embeddings_visualization.png: PCA visualization of embeddings")


if __name__ == "__main__":
    main() 