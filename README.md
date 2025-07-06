# Deep Set Autoencoder with PyTorch

A clean, modular implementation of a Deep Set autoencoder that learns embeddings from sets of data points. The model is trained using Chamfer distance loss and can handle both numerical and categorical features.

## Overview

This project implements a set-based neural network (Deep Set) autoencoder that:
- Takes sets of data points as input (permutation invariant)
- Learns meaningful embeddings that capture set properties
- Can reconstruct sets from embeddings
- Handles both numerical and categorical features
- Uses Chamfer distance for set-to-set comparison

The demo includes a synthetic dataset where each set contains points from 3 different distributions, allowing us to visualize how the model learns to cluster sets based on their composition.

## Project Structure

```
BasicNeuralNets/
├── main.py                 # Main script - complete pipeline
├── generate_dummy_data.py  # Data generation utilities
├── dataset.py             # Dataset handling and dataloaders
├── model.py               # Deep Set autoencoder architecture
├── scorer.py              # Loss functions (Chamfer distance)
├── environment.yml        # Conda environment specification
└── README.md              # This documentation
```

## Architecture

The Deep Set architecture consists of:
1. **Encoder**: 
   - φ network: Processes each element in the set independently
   - Aggregation: Permutation-invariant operation (mean/sum/max)
   - ρ network: Processes the aggregated representation to produce embedding

2. **Decoder**: 
   - Maps embeddings back to sets of points
   - Generates all set elements simultaneously

## Features

- **Permutation Invariant**: Order of elements in input sets doesn't matter
- **Categorical Support**: Automatic embedding of categorical features
- **Flexible Architecture**: Configurable hidden dimensions, layers, and aggregation methods
- **Chamfer Distance Loss**: Appropriate loss function for set reconstruction
- **In-Memory Data Generation**: No disk I/O required
- **Complete Pipeline**: Single script runs everything

## Installation

1. **Create conda environment:**
```bash
conda env create -f environment.yml
```

2. **Activate environment:**
```bash
conda activate basic-neural-nets
```

## Quick Start

Run the complete pipeline with a single command:

```bash
python main.py
```

This will:
1. Generate synthetic data in memory
2. Initialize the dataset and dataloaders
3. Create and train the Deep Set autoencoder
4. Analyze and visualize the learned embeddings
5. Save the trained model and visualization

## Data Structure

### Input Format
- **Shape**: `(batch_size, set_size, input_dim)`
- **Example**: `(32, 20, 5)` = 32 sets, 20 points each, 5 dimensions per point
- **Permutation Invariant**: Order of points within each set doesn't matter

### Synthetic Data
Each set contains points from 3 different distributions:
- **Distribution 1**: Centered at `(0, 0, 0, 0, 0)` with std=1.0
- **Distribution 2**: Centered at `(5, 5, 5, 5, 5)` with std=1.2  
- **Distribution 3**: Centered at `(-5, 5, 5, 5, 5)` with std=0.8

Each set is labeled based on which distribution contributes the most points (60-80%).

## Configuration

The main configuration is in `main.py`:

```python
config = {
    'num_sets': 5000,           # Number of sets to generate
    'set_size': 20,             # Points per set
    'input_dim': 5,             # Dimensions per point
    'categorical_features': 0,  # Number of categorical features
    'embedding_dim': 32,        # Embedding dimension
    'hidden_dim': 128,          # Hidden layer dimension
    'num_layers': 3,            # Number of layers
    'aggregation': 'mean',      # Aggregation method
    'batch_size': 64,           # Batch size
    'learning_rate': 1e-3,      # Learning rate
    'num_epochs': 30,           # Training epochs
    'seed': 42                  # Random seed
}
```

## Usage Examples

### Basic Usage

```python
from model import DeepSetAutoencoder
from generate_dummy_data import generate_dummy_data
from dataset import create_dataloaders

# Generate data
data, labels = generate_dummy_data(num_sets=1000, set_size=20, input_dim=5)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(data, labels)

# Create model
model = DeepSetAutoencoder(
    input_dim=5,
    embedding_dim=32,
    set_size=20,
    hidden_dim=128
)

# Forward pass
for batch_data, batch_labels in train_loader:
    reconstructed, embeddings = model(batch_data)
    break
```

### With Categorical Features

```python
# Generate data with categorical features
data, labels = generate_dummy_data(
    num_sets=1000,
    set_size=20,
    input_dim=7,  # 5 numerical + 2 categorical
    categorical_features=2,
    num_categories=5
)

# Create model with categorical support
model = DeepSetAutoencoder(
    input_dim=7,
    categorical_dims=[5, 5],  # 5 categories for each feature
    embedding_sizes=[3, 3],   # Embedding dimensions
    ...
)
```

## Model Performance

The model should learn to:
1. Cluster sets based on their dominant distribution
2. Reconstruct sets with low Chamfer distance
3. Generate meaningful 2D visualizations via PCA showing 3 distinct clusters

Typical metrics after training:
- **Silhouette Score**: ~0.6-0.8
- **K-means Clustering Accuracy**: ~0.8-0.95
- **Reconstruction Error**: Chamfer distance < 1.0

## Output Files

After running `main.py`, you'll get:
- `best_model.pth`: Trained model weights
- `embeddings_visualization.png`: PCA visualization of learned embeddings

## Customization

### Change Aggregation Method

```python
model = DeepSetAutoencoder(
    aggregation='sum',  # Options: 'mean', 'sum', 'max'
    ...
)
```

### Adjust Model Capacity

```python
model = DeepSetAutoencoder(
    hidden_dim=256,     # Increase hidden dimensions
    num_layers=5,       # Add more layers
    embedding_dim=64,   # Larger embeddings
    ...
)
```

### Modify Data Generation

```python
# Generate more complex data
data, labels = generate_dummy_data(
    num_sets=10000,     # More sets
    set_size=50,        # Larger sets
    input_dim=10,       # Higher dimensions
    categorical_features=3,  # More categorical features
    num_categories=10
)
```

## Key Components

### `generate_dummy_data.py`
- `generate_dummy_data()`: Creates synthetic sets with mixed distributions
- `get_feature_info()`: Provides feature information for model construction

### `dataset.py`
- `SetDataset`: PyTorch dataset for set-based data
- `create_dataloaders()`: Creates train/val/test splits

### `model.py`
- `DeepSetEncoder`: Encodes sets to embeddings
- `DeepSetDecoder`: Decodes embeddings back to sets
- `DeepSetAutoencoder`: Complete autoencoder with categorical support

### `scorer.py`
- `ChamferLoss`: Chamfer distance loss for set reconstruction
- `chamfer_distance()`: Raw Chamfer distance computation

### `main.py`
- Complete training pipeline
- Embedding analysis and visualization
- Model evaluation and saving

## Troubleshooting

**Issue**: Poor clustering in embeddings
- Try increasing `embedding_dim`
- Train for more epochs (`num_epochs`)
- Adjust the distribution parameters in `generate_dummy_data.py`

**Issue**: High reconstruction error
- Increase model capacity (`hidden_dim`, `num_layers`)
- Check if `set_size` in model matches data
- Reduce learning rate

**Issue**: Training instability
- Reduce `learning_rate`
- Increase `batch_size`
- Add gradient clipping

## Future Extensions

Potential improvements and extensions:
- Attention-based aggregation instead of mean/sum/max
- Variational autoencoder version
- Support for variable-size sets
- More sophisticated categorical embeddings
- Additional loss functions (e.g., Earth Mover's Distance)
- Real-world dataset support

## License

This project is provided as-is for educational and research purposes. 