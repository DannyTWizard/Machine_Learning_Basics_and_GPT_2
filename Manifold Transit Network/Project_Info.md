# Manifold Transit Network (MNT)

A novel neural network architecture that represents computation as particle dynamics in a learned potential field. Instead of traditional layer-wise transformations, inputs are encoded as particles that evolve under forces from learnable attractors.

## Core Concept

Traditional neural networks transform inputs through sequential matrix multiplications. MNT takes a different approach:

1. **Encode** inputs into a low-dimensional latent space
2. **Simulate** particle dynamics where the encoded representation moves under influence of learnable attractors
3. **Decode** the final particle position to produce outputs

The dynamics follow a force law derived from a Graph Laplacian formulation:

```
F(x) = -sum_k w_k * (x - c_k) / ||x - c_k||
x' = x + eta * F(x)
```

Where:
- `x` is the particle (encoded input)
- `c_k` are learnable attractor positions
- `w_k` are weights derived from a potential function
- `eta` is the step size

## Architecture

```
Input (784) --> Encoder --> Latent Space (N) --> FLAP Layers --> Decoder --> Output (10)
                              |                      |
                              |    Attractor Field   |
                              |    (K attractors)    |
                              +----------------------+
```

### Components

- **Encoder**: Linear projection from input dimension to latent space
- **FLAP Layer** (Force-based Learned Attractor Potential): Computes forces from attractors and updates particle position
- **Decoder**: Linear projection from latent space to output dimension
- **Attractors**: Learnable parameters representing fixed points in latent space

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- PyTorch
- torchvision
- matplotlib
- numpy

## Usage

### Training the MNT model on MNIST

```bash
python src/MNT_v0/MNT_flap.py
```

### Training the baseline control model

```bash
python src/MNT_v0/MNT_control.py
```

## File Structure

```
Manifold-Transit-Network/
├── src/
│   └── MNT_v0/
│       ├── MNT_flap.py      # Main MNT implementation
│       └── MNT_control.py   # Baseline model for comparison
├── requirements.txt
└── README.md
```

## Key Hyperparameters

| Parameter | Description | Current Value |
|-----------|-------------|---------------|
| `N` | Latent space dimension | 5 |
| `num_attractors` | Number of attractor points | 30 |
| `eta` | Step size for dynamics | 2.3 |
| `iterations` | Update steps per layer | 1 |
| `num_attractor_layers` | Number of FLAP layers | 1 |

## Training Details

- **Optimizer**: Adam with separate parameter groups
- **Weight decay**: 0.01 for encoder/decoder, 0.0 for attractors
- **Gradient clipping**: 1.0 for encoder/decoder, 0.5 for attractors
- **Loss**: Negative Log Likelihood (NLLLoss)

## Theoretical Background

The architecture draws inspiration from:
- Neural manifold learning in neuroscience
- N-body gravitational simulations
- Hopfield networks (but with explicit geometric attractor coordinates rather than implicit weight matrices)
- Dynamical systems operating at the "edge of chaos"

The Graph Laplacian formulation enables efficient computation of forces from all attractors simultaneously, and the learnable potential function allows the network to discover optimal force profiles during training.

## License

See LICENSE file for details.
