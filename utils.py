import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.widgets import Slider
import sys
import os
from datetime import datetime

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


def show_grid(images, titles=None, shape=(28, 28), cols=5, cmap='gray', save_dir=None, filename_prefix=None):
    """Display a grid of images or save it to `save_dir` when provided.

    If `save_dir` is None the function will call `plt.show()` as before.
    Otherwise it will save the figure as a PNG with a timestamped filename
    using the optional `filename_prefix`.
    """
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(2 * cols, 2 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.reshape(shape), cmap=cmap, interpolation='nearest')
        plt.axis('off')
        if titles is not None:
            # guard against titles shorter than images
            if i < len(titles):
                plt.title(titles[i], fontsize=8)
    plt.tight_layout()

    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            # if creation fails, fall back to show
            plt.show()
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        prefix = filename_prefix or 'grid'
        filename = f"{prefix}_{ts}.png"
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {path}")
        plt.close()
    else:
        plt.show()



def visualize_originals(X_test, n_show=10, save_dir=None):
    originals = X_test[:n_show]
    print("Displaying original test images...")
    show_grid(originals, titles=[f'orig {i}' for i in range(n_show)], save_dir=save_dir, filename_prefix='originals')


def visualize_reconstructions(rbm, X_test, n_show=10, save_dir=None):
    print("Reconstructing test images through RBM...")
    originals = X_test[:n_show]
    hidden = rbm.run_visible(originals)
    reconstructed = rbm.run_hidden(hidden)
    recon_display = (reconstructed > 0.5).astype(np.float32)
    show_grid(recon_display, titles=[f'recon {i}' for i in range(n_show)], save_dir=save_dir, filename_prefix='reconstructions')


def visualize_daydream_samples(rbm, num_samples=10, mode="normal", save_dir=None):
    print("Generating new samples (daydream)...")
    samples = rbm.daydream(num_samples=num_samples, burn_in=6000, thinning=6000, mode=mode)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    show_grid(samples[:10], titles=[f'sample {i}' for i in range(min(10, samples.shape[0]))], save_dir=save_dir, filename_prefix='daydream')


def visualize_impaired_reconstruction(rbm, X_test, mode="normal", save_dir=None):
    print("Running impaired (half-masked) daydream tests...")
    example = X_test[0]
    img = example.reshape(28, 28).copy()
    impaired = img.copy()
    impaired[:, 14:] = 0.0
    impaired_flat = impaired.reshape(1, -1)

    # show original and impaired
    show_grid([img, impaired], titles=['original', 'impaired'], cols=2, save_dir=save_dir, filename_prefix='impaired_pair')

    # deterministic reconstruction
    hid = rbm.run_visible(impaired_flat)
    recon_prob = rbm.run_hidden(hid)
    recon_det = (recon_prob > 0.5).astype(np.float32).reshape(28, 28)
    show_grid([impaired, recon_det], titles=['impaired', 'recon_from_impaired'], cols=2, save_dir=save_dir, filename_prefix='impaired_recon')

    # daydream continuation
    chain_samples = rbm.daydream(
        num_samples=10, initial_visible=impaired_flat,
        burn_in=6000, thinning=1000, mode=mode
    )
    if chain_samples.ndim == 1:
        chain_samples = chain_samples.reshape(1, -1)
    show_grid(chain_samples[:10], titles=[f'imputed {i}' for i in range(chain_samples.shape[0])], cols=2, save_dir=save_dir, filename_prefix='imputed_chain')


def visualize_hidden_weights(rbm, n_filters=64, n_cols=8, shape=(28, 28), save_dir=None):
    print("Displaying first 64 learned hidden-unit weights (as images)")
    vis_hidden = rbm.weights[1:, 1:]
    imgs = []
    for i in range(min(vis_hidden.shape[1], n_filters)):
        imgs.append(vis_hidden[:, i].reshape(shape))
    show_grid(imgs, cols=n_cols, titles=[f'w{i}' for i in range(len(imgs))], shape=shape, save_dir=save_dir, filename_prefix='hidden_weights')


def visualize_particle_trajectories(trajectories, save_dir=None, filename_prefix='trajectories',
                                   shape=(28, 28), cols=10, cmap='gray', interval=300, writer='pillow'):
    """Create an animation showing particle samples at each timestep.

    If save_dir is None, show an interactive window with a slider to drag time steps.
    If save_dir is set, save as GIF (no slider).
    """
    arr = np.asarray(trajectories)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], 1, arr.shape[1])
    if arr.ndim != 3:
        raise ValueError("trajectories must be shape (T, P, V) or (T, V)")

    T, P, V = arr.shape
    rows = int(np.ceil(P / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.array(axes).reshape(-1)

    def draw_frame(frame):
        for p in range(P):
            ax = axes[p]
            img = arr[frame, p].reshape(shape)
            ax.imshow(img, cmap=cmap, interpolation='nearest')
            ax.set_title(f'p{p}', fontsize=8)
            ax.axis('off')
        for k in range(P, axes.size):
            axes[k].clear()
            axes[k].axis('off')
        fig.suptitle(f'Timestep: {frame+1}/{T}', fontsize=14)
        fig.canvas.draw_idle()

    if save_dir:
        # Save as GIF (no slider)
        def init():
            for ax in axes:
                ax.axis('off')
            return axes
        def update(frame):
            draw_frame(frame)
            return axes
        ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, interval=interval, blit=False)
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{filename_prefix}_{ts}.gif"
        path = os.path.join(save_dir, filename)
        try:
            if writer == 'pillow':
                pw = PillowWriter(fps=1000 / interval)
                ani.save(path, writer=pw)
            else:
                ani.save(path)
            plt.close(fig)
            print(f"Saved particle trajectories animation to: {path}")
            return path
        except Exception as exc:
            print("Failed to save animation, falling back to show():", exc)
        plt.show()
        return None
    else:
        # Interactive slider
        plt.subplots_adjust(bottom=0.18)
        ax_slider = plt.axes([0.2, 0.08, 0.6, 0.04])
        slider = Slider(ax_slider, 'Timestep', 1, T, valinit=1, valstep=1)
        draw_frame(0)

        def on_slider(val):
            frame = int(val) - 1
            draw_frame(frame)

        slider.on_changed(on_slider)
        plt.show()
        return None

def visualize_energy_landscape(rbm, X_sample, dim_idx=(0, 1), grid_size=50, v_current=None, save_dir=None):
    """
    Visualize a 2D energy landscape of the RBM by varying two visible dimensions.

    Parameters
    ----------
    rbm : trained RBM instance (must have rbm.weights)
    X_sample : array-like, shape (num_visible,)
        Reference visible vector to fix all other dimensions.
    dim_idx : tuple of two ints
        Indices of visible units to vary.
    grid_size : int
        Resolution of grid.
    v_current : optional
        Current visible vector (e.g., sampled state) to highlight.
    """
    W = rbm.weights  # shape (num_visible+1, num_hidden+1)
    n_v = W.shape[0] - 1
    n_h = W.shape[1] - 1

    # prepare grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    E = np.zeros_like(X)

    # fix other dimensions
    base = X_sample.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            v = base.copy()
            v[dim_idx[0]] = X[i, j]
            v[dim_idx[1]] = Y[i, j]
            v_bias = np.insert(v, 0, 1)  # add bias
            # hidden probabilities
            # h_prob = 1 / (1 + np.exp(-v_bias @ W))
            # # energy E(v,h*) = -v^T W h_prob (mean-field energy)
            # E[i, j] = -np.dot(v_bias, np.dot(W, h_prob))

            h_prob = 1 / (1 + np.exp(-v_bias @ W))
            h_state = (np.random.rand(h_prob.size) < h_prob).astype(float)
            E[i, j] = -np.dot(v_bias, np.dot(W, h_state))

    # plot
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, E, levels=50, cmap='viridis')
    plt.colorbar(contour, label="Energy E(v)")
    plt.xlabel(f"Visible unit {dim_idx[0]}")
    plt.ylabel(f"Visible unit {dim_idx[1]}")
    plt.title("RBM Energy Landscape (2D slice)")

    # optionally mark current sample
    if v_current is not None:
        plt.scatter(v_current[dim_idx[0]], v_current[dim_idx[1]],
                    color='red', s=80, edgecolors='white', label='current point')
        plt.legend()

    plt.tight_layout()
    # save if caller requested
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            fname = f"energy_landscape_{dim_idx[0]}_{dim_idx[1]}_{ts}.png"
            path = os.path.join(save_dir, fname)
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {path}")
            plt.close()
            return
        except Exception:
            pass

    plt.show()