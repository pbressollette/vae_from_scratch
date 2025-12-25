# Variational Autoencoder (VAE) from Scratch

Implementation of a Variational Autoencoder (VAE) in PyTorch, trained on the MNIST dataset.

## Motivation

VAEs are powerful generative models that combine deep learning with probabilistic inference. I implemented this model from scratch to understand the mathematics behind variational inference and the reparameterization trick, and to explore the structure of learned latent representations.

## Dataset

I used the dataset [MNIST](http://yann.lecun.com/exdb/mnist/).

"The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image of 28x28 pixels."

## Installation & Usage

The project can be run locally or on any platform with Python 3.x and PyTorch support.

You are free to modify the hyperparameters directly in the notebook. The main parameters are:
- `LATENT_DIM`: dimension of the latent space (default: 20)
- `BATCH_SIZE`: number of images per batch (default: 64)
- `LEARNING_RATE`: optimizer learning rate (default: 1e-3)
- `NUM_EPOCHS`: number of training epochs (default: 10)

Once you have the desired configuration, install the packages and run the notebook.
```bash
# packages installation
pip install torch torchvision matplotlib numpy jupyter

# run the notebook
jupyter notebook vae_from_scratch.ipynb
```

## Results

### Training Curves

After training for 10 epochs, the model converged successfully:

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1     | 7267.97    | 7022.17  |
| 5     | 6773.17    | 6675.44  |
| 10    | 6629.47    | 6563.28  |

One can see that both training and validation losses decrease steadily. The validation loss remains lower than the training loss throughout training, indicating good generalization with no overfitting.

### Architecture

The VAE consists of three main components:

**Encoder**: Convolutional layers that compress 28x28 images into a 20-dimensional latent representation, outputting both mean (μ) and log-variance (log σ²).

**Reparameterization**: Samples z = μ + σ × ε where ε ~ N(0,1), enabling backpropagation through stochastic nodes.

**Decoder**: Transposed convolutional layers that reconstruct 28x28 images from latent vectors.

**Loss Function**: Combination of reconstruction loss (binary cross-entropy) and KL divergence regularization.

### Capabilities

The trained model successfully:
- Reconstructs input images with high fidelity (slight blur is expected due to the probabilistic nature of VAEs)
- Generates novel digit images by sampling from N(0,1) in the latent space
- Performs smooth interpolation between different digits in latent space

### Examples Visualization

Here are examples of the model's capabilities:

**Reconstruction**: The model accurately reconstructs input digits while preserving their essential features. Some fine details are slightly blurred, which is characteristic of VAEs learning distributions rather than exact mappings.

**Generation**: Sampling random vectors from N(0,1) and decoding them produces realistic-looking digits.

**Interpolation**: The model can smoothly morph between different digits (e.g., 6 → 8) by linearly interpolating in the latent space, demonstrating that it has learned a continuous and structured representation.