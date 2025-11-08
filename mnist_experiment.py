"""
mnist_experiment.py

Small experiment script that loads binary (binarized) MNIST data,
trains the RBM from this repo on a small subset, shows reconstructions
and generates a few samples.

This script tries to use TensorFlow's Keras MNIST loader first, and
falls back to scikit-learn's fetch_openml if TensorFlow isn't available.

The script intentionally uses a small number of epochs / examples by
default so it runs quickly for experimentation. Tune parameters as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from rbm import RBM


def load_mnist_binarized(n_train=2000, n_test=500, random_binarize=True, seed=123):
    """Load MNIST and return binarized train/test arrays (values 0/1).

    Binarization strategy:
    - If random_binarize=True: treat each pixel intensity (0..255 or 0..1)
      as the probability p of a Bernoulli and sample from it.
    - Else: use deterministic threshold at 0.5.

    Returns X_train, X_test shaped (n_samples, 784) with dtype float32 (0/1).
    """
    # Load MNIST using PyTorch/torchvision. This is the preferred path for
    # this experiment; ensure `torch` and `torchvision` are installed.
    try:
        import torch
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError(
            "Unable to import torch/torchvision. Install `torch` and `torchvision` to run this experiment."
        ) from exc

    # Download (if needed) and load train+test
    train = datasets.MNIST(root='./data', train=True, download=True)
    test = datasets.MNIST(root='./data', train=False, download=True)

    # `train.data` / `test.data` are torch tensors with shape (N, 28, 28)
    x_train = train.data.numpy().astype(np.float32) / 255.0
    x_test = test.data.numpy().astype(np.float32) / 255.0
    x = np.vstack([x_train, x_test])

    # For reproducibility
    rng = np.random.RandomState(seed)

    # Shuffle
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    x = x[indices]

    # Select subset
    n_total = n_train + n_test
    if n_total > x.shape[0]:
        raise ValueError("Requested more samples than available in MNIST")
    x = x[:n_total]

    if random_binarize:
        # For each pixel, sample Bernoulli(p = pixel intensity)
        x_bin = rng.binomial(1, x).astype(np.float32)
    else:
        x_bin = (x > 0.5).astype(np.float32)

    X_train = x_bin[:n_train].reshape(n_train, -1)
    X_test = x_bin[n_train:n_train + n_test].reshape(n_test, -1)
    return X_train, X_test


def show_grid(images, titles=None, shape=(28, 28), cols=5, cmap='gray'):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(2 * cols, 2 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.reshape(shape), cmap=cmap, interpolation='nearest')
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    print("Loading and binarizing MNIST (this may take a moment the first time)...")
    # Use deterministic binarization for clearer training signal
    X_train, X_test = load_mnist_binarized(n_train=2000, n_test=200, random_binarize=False)

    num_visible = X_train.shape[1] 
    num_hidden = 128

    print(f"Training RBM with num_visible={num_visible}, num_hidden={num_hidden}")
    rbm = RBM(num_visible=num_visible, num_hidden=num_hidden)

    # Train: increase epochs and use minibatches for stability
    # Use CD-k with k=5 for a better negative-phase approximation
    rbm.train(X_train, max_epochs=200, learning_rate=0.01, batch_size=64, cd_k=1)

    # Show original test examples
    n_show = 10
    originals = X_test[:n_show]
    print("Displaying original test images...")
    show_grid(originals, titles=[f'orig {i}' for i in range(n_show)])

    # Reconstruct each original via RBM: visible -> hidden -> visible
    print("Reconstructing test images through RBM...")
    hidden = rbm.run_visible(originals)
    reconstructed = rbm.run_hidden(hidden)

    # The RBM returns probabilities (floating 0..1). Threshold for display
    recon_display = (reconstructed > 0.5).astype(np.float32)
    show_grid(recon_display, titles=[f'recon {i}' for i in range(n_show)])

    # Generate new samples using daydream (Gibbs chain)
    print("Generating new samples (daydream)...")
    # Run a long Gibbs chain: use burn-in then take a few samples
    samples = rbm.daydream(num_samples=10, burn_in=3000, thinning=50)
    # samples shape: (num_samples, num_visible) when batch==1
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    show_grid(samples[:10], titles=[f'sample {i}' for i in range(min(10, samples.shape[0]))])

    # Visualize the first 64 learned hidden weights as 28x28 filters
    try:
        def show_weights(weights, n_cols=8, shape=(28,28)):
            # weights: (num_visible, num_hidden)
            n_filters = min(weights.shape[1], 64)
            imgs = []
            for i in range(n_filters):
                imgs.append(weights[:, i].reshape(shape))
            show_grid(imgs, cols=n_cols, titles=[f'w{i}' for i in range(n_filters)], shape=shape)

        # RBM weights include bias row/col; visible-to-hidden weights are weights[1:,1:]
        vis_hidden = rbm.weights[1:, 1:]
        print("Displaying first 64 learned hidden-unit weights (as images)")
        show_weights(vis_hidden)
    except Exception:
        # Non-fatal: show_weights is optional
        pass

    # --- Impaired (half-masked) daydream test ---
    print("Running impaired (half-masked) daydream tests...")
    # Take a single test example and mask half of the pixels (right half)
    example = X_test[0]
    img = example.reshape(28, 28).copy()
    impaired = img.copy()
    # Mask right half
    impaired[:, 14:] = 0.0
    impaired_flat = impaired.reshape(1, -1)

    # Show original and impaired
    show_grid([img, impaired], titles=['original', 'impaired'], cols=2)

    # Deterministic reconstruction of impaired input (one pass V->H->V)
    hid = rbm.run_visible(impaired_flat)
    recon_prob = rbm.run_hidden(hid)
    recon_det = (recon_prob > 0.5).astype(np.float32).reshape(28, 28)
    show_grid([impaired, recon_det], titles=['impaired', 'recon_from_impaired'], cols=2)

    # Daydream starting from impaired input: short burn-in and some thinning
    chain_samples = rbm.daydream(num_samples=10, initial_visible=impaired_flat, burn_in=3000, thinning=50)
    # If batch==1, chain_samples shape: (num_samples, num_visible)
    if chain_samples.ndim == 1:
        chain_samples = chain_samples.reshape(1, -1)
    # Display the generated samples
    show_grid(chain_samples, titles=[f'imputed {i}' for i in range(chain_samples.shape[0])], cols=5)

    print("Done.")


if __name__ == '__main__':
    main()
